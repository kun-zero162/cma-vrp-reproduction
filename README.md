# CMA-VRP reproduction

Reproduction of **"Entity Graph Alignment and Visual Reasoning for Multimodal Fake News
Detection"**, Guoyi Li, Die Hu, Xiaomeng Fu, Qirui Tang, Yulei Wu, Xiaodan Zhang,
Honglei Lyu — *ACM Multimedia 2025*, DOI [10.1145/3746027.3754591](https://doi.org/10.1145/3746027.3754591).

The authors did not release code. This subproject rebuilds the method from the paper text
(`../paper.txt`, extracted from `3746027.3754591.pdf`) and reuses the neighbouring
`Expert/` (EASE) project for the training scaffolding — logging, optimizer/scheduler
construction, recorder and metrics — via `ease_bridge.py`, which loads those functions
*by path* so `configs`/`models`/`datasets` keep resolving to this package.

**Status: the full pipeline runs end to end** (graph construction → cache → train → test →
robustness), and every loss term in Eq.13 receives gradient. **It is not yet a numerical
reproduction of the paper's tables**, because the three benchmarks are not available here
(see *What still blocks a real reproduction*).

---

## Layout

| Path | Paper |
|---|---|
| `preprocess/text_graph.py` | Sec.3.1.1 textual entity graph `G_t` (spaCy NER + dependency edges, global node, 4 generic type nodes) |
| `preprocess/visual_graph.py` | Sec.3.1.1 visual entity graph `G_v` (DETR boxes/labels, ResNet50 region features, geometric predicates) |
| `preprocess/vrp.py` | Sec.3.1.2 dual-layer truth-driven CoT Visual Reasoning Prompting → `VRF_t`, `VRF_v` |
| `preprocess/entity_matching.py` | Sec.3.2.1 (i) entity matching → cross adjacency `A_tv` |
| `preprocess/build_dataset.py` | orchestrates the above + BERT node encoding → `cache/{ds}/{split}.npz` |
| `models/contrastive.py` | Sec.3.2.1 Eq.1-4 graph contrastive learning |
| `models/gnn.py` | Sec.3.2.2 Eq.5-7 Uni-Att + cross-modal gating fusion (L=2) |
| `models/cycle_align.py` | Sec.3.3 Eq.8-9 VSCA (Sinkhorn-OT + WGAN critic), Eq.10-12 ACCIL |
| `models/CMAVRP.py` | Sec.3.4 Eq.13 `x̂ = o ⊕ c_m`, `L = L_y + λL_cl + βL_adv` |
| `datasets/graph_dataset.py` | pickle-free cache reader |
| `engine.py`, `train.py`, `test.py`, `robustness.py` | Eq.13 training loop, evaluation, Sec.4.3 robustness study |
| `tools/unit_forward.py` | shape/gradient/ablation self-check on random tensors (no data needed) |
| `tools/make_synthetic.py` | synthetic Twitter-format stand-in so the pipeline can be exercised |
| `tools/ablation_table.py` | runs `full` + all six Table-2 ablations per dataset and aggregates `results/ablation_{ds}.{md,csv,json}` |

## Quick start (CPU, no GPU or dataset required)

```bash
# 0. one-time environment. The repo's .venv is Python 3.14 and has no torch wheels at
#    all; use 3.10 and the CPU build.
py -3.10 -m venv ../.venv310
../.venv310/Scripts/python.exe -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
../.venv310/Scripts/python.exe -m pip install -r requirements.txt
../.venv310/Scripts/python.exe -m spacy download en_core_web_sm

# 1. sanity-check the model alone (no data, no weights)
../.venv310/Scripts/python.exe tools/unit_forward.py

# 2. weights. HF's xet transport 404s anonymously here, so set HF_HUB_DISABLE_XET=1
../.venv310/Scripts/python.exe -c "from huggingface_hub import snapshot_download as s; \
  s('google-bert/bert-base-uncased', local_dir='bert/bert-base-uncased'); \
  s('facebook/detr-resnet-50', local_dir='bert/detr-resnet-50')"

# 3. data + cache
../.venv310/Scripts/python.exe tools/make_synthetic.py --datasets twitter
../.venv310/Scripts/python.exe -m preprocess.build_dataset --datasets twitter \
    --splits train val test --device cpu --spacy_model en_core_web_sm --vrp_backend mock

# 4. train / test / robustness
../.venv310/Scripts/python.exe train.py --datasets twitter --cpu --epochs 12 --batch_size 8
../.venv310/Scripts/python.exe test.py --datasets twitter --cpu
../.venv310/Scripts/python.exe robustness.py --datasets twitter --cpu
```

`../preflight_cmavrp.py` reports which of steps 0-3 are still missing and prints the
exact commands for this machine.

`train.sh` is the full-protocol version (3 datasets × 50 epochs + all six Table-2
ablations) intended for a GPU box with the real benchmarks in place.

## Hyperparameters (Sec.4)

`lr=1e-3`, `StepLR(step_size=5, gamma=0.5)`, 50 epochs with early stop 5, `λ=0.5`,
`β=0.1`, batch 32, `p_m=0.2`, θ init 0.5 (learnable, sigmoid-parameterised),
τ=0.07, 2 fusion layers, `gnn_dim=300`, BERT-base-uncased node initialisation,
spaCy `en_core_web_trf` parsing, Mistral-7B for textual and LLaVA-1.6-13B for visual
entities. All in `configs/config_setting.py`.

## Verified

* `tools/unit_forward.py` — all checks pass: dense batch, sparse/padding batch
  (including a sample with *no* visual nodes and one with *no* text nodes), the θ
  gradient, and each of the six Table-2 ablations. Each switch that removes a loss term
  is asserted to drive **that term to exactly zero** (`L_inter`, `L_intra`, `L_cycle`,
  `L_D0`), so the switches gate the objective rather than being cosmetic. No non-finite
  gradients; 0/149 parameters without gradient in the full model.
* **Correction (Nov 2026):** an earlier revision of this file claimed all six ablations
  zeroed their term. That was false for `use_accil`. `forward()` guarded the block with
  `if use_accil or use_vsca` but then called `self.accil(...)` unconditionally, so with
  VSCA enabled (the default) disabling ACCIL changed nothing — the `no_use_accil` row of
  both tables was the full model re-measured, bit for bit. Each term is now gated
  independently and `unit_forward.py` asserts `L_D0 == 0` under `-use_accil`, which is the
  check that would have caught it originally.
* `train.py` on 60 synthetic posts: `loss_total` 9.45 → 2.93, adversarial term 62 → 5.6 as
  the gradient-reversal ramp progresses, val AUC 0.98 at epoch 0, early stop fires
  correctly, checkpoints written. 8.07 M parameters, 28 s for 6 epochs on CPU.
* `test.py` and `robustness.py` load the checkpoint **strictly** (no key mismatch) and
  produce metrics, a t-SNE figure and a robustness table.
* `robustness.py` sweeps behave as expected: the Sinkhorn-Wasserstein distance between the
  aligned node clouds rises with feature noise (155 → 191) and falls with visual-node
  occlusion (155 → 112); AUC degrades as the perturbation grows.

## Table 2 ablations — what these numbers do and do not show

`python tools/ablation_table.py --datasets twitter weibo --epochs 12 --early_stop 5
--batch_size 8 --cpu` writes `results/ablation_<dataset>.{md,csv,json}` (one row per
config, delta vs `full`) plus a per-run transcript in `logs/ablation/`.

The tables in the repo right now were produced on: the synthetic Twitter stand-in and its
Chinese twin (60 train / 16 val / 16 test each — the Weibo cache really does go through
the zh pipeline; its entity nodes are `消防栓`, `人群`, ...), `vrp_backend=mock`, 12
epochs, batch 8, seed 3759, CPU. **16 test samples per dataset**, so every
accuracy/mac_f1 value moves in steps of 1/16 = 0.0625 and one flipped prediction is a
6-point change.

What this *does* establish, about the implementation:

* All six switches change what is trained, and the ones that remove a loss term are
  asserted to zero exactly that term (`unit_forward.py`).
* `use_accil` was not one of them until the fix above. With the bug, `no_use_accil` was
  bit-for-bit `full`; after it, the row is a genuine degradation (Twitter AUC
  0.8833 → 0.5500, Weibo accuracy 0.8125 → 0.3750, `recall_real` → 0.0000 on both). The
  model collapses to predicting *fake* for everything, which is what removing the
  invariant-prediction columns of `x̂` (Eq.13) should do on a set this small.
* `no_use_gcl_inter` and `no_use_gcl_intra` land on identical metric rows on both
  datasets, but are genuinely different models — different checkpoint hashes, different
  training loss (epoch 0: 8.31 vs 7.85). Two different ablations producing the same
  16-sample confusion matrix is a warning about the sample size, not evidence that the
  two terms are interchangeable.

What this does **not** establish, and must not be quoted as a result:

* Removing CMGAF *improves* both datasets (Twitter +0.117, Weibo +0.283 AUC) and removing
  the inter/intra graph losses improves accuracy. The synthetic stand-in is solvable from
  text alone, so the fusion and alignment machinery has nothing useful to add here — this
  measures the stand-in data, not the paper's components.
* `no_use_vrf` reaching AUC 1.0000 on Weibo is meaningless: with `vrp_backend=mock` the
  VRFs are deterministic templates rather than LLM reasoning, so ablating them removes
  boilerplate, not evidence.

A Table-2 claim in the paper's sense needs the real benchmarks, a real VRP backend, and
multi-seed runs — see below.

## Deliberate deviations

These are substitutions forced by missing assets, each isolated so the paper's version can
be dropped back in:

1. **SGG.** The paper uses a trained scene-graph generator (Mask-RCNN + relation head).
   Here DETR supplies boxes/labels and predicates are derived from box geometry
   (`visual_graph._box_relations`). The node/edge vocabulary and tensor layout are
   unchanged, so only `_propose` + `_box_relations` need replacing.
2. **VRP.** `vrp.py` has three backends: `mock` (deterministic template reasoning, used for
   the smoke test), `hf` (local Mistral/LLaVA via transformers — needs ≥24 GB VRAM), and
   `agent` (the repo's existing EASE `Agent/` LLM plumbing, i.e. an API). This machine has
   a 4 GB GTX 1650, so `mock`/`agent` are the only usable options.
3. **OCR.** The paper's Rosetta is not public, so `ocr_backend='none'` by default;
   `easyocr` is available if wanted.
4. **Entity matching.** `string` matching approximates the LVLM's recognition step;
   `embedding` mode is implemented but unused. Global `v_g` and generic type nodes are
   excluded from string matching — the visual global node embeds the whole caption, which
   otherwise makes every textual entity match it.
5. **Adversarial training.** Eq.9/Eq.12 are min-max games, not a single scalar objective.
   They are driven through EASE's `ReverseLayerF` gradient-reversal layer with a DANN ramp
   (`set_progress`), so one optimizer trains both players. This is why `L_D1 = -J`.
6. **`timm` is required** (not optional): transformers 5.x routes DETR's conv backbone
   through `TimmBackbone`.
7. **Synthetic data.** `tools/make_synthetic.py` renders COCO-labelled scenes and writes
   captions that contradict the drawn objects for `label=1`, so the smoke test exercises
   real cross-modal misalignment rather than noise. **Scores from it are meaningless as
   evidence about the method** — 92 samples, text-only signal, and DETR cannot detect flat
   coloured shapes (which is why `visual_graph` has a 2×2 grid proposal fallback that only
   triggers when the detector finds nothing).

## Paper-faithfulness notes

* Eq.3 supervises a **[B,B] cross-similarity logit matrix**, not per-sample embeddings:
  `Ỹ = θY + (1−θ)S` with `Y = I` (the true visual match for text row *i* is column *i*) and
  `S = softmax_j(sim/τ)`. An earlier draft wrongly applied it to `[B, d]` logits.
* Eq.4 is SimCLR at the **sample** level over the fused `g(·)` representation — anchor =
  original pair, positive = entity-masked graph, negatives = modality-mixed graphs — not a
  per-modality same-label InfoNCE.
* `L_intra` is legitimately **negative**: Eq.4's denominator excludes the positive term, so
  `-log(exp(pos)/Σexp(neg))` has no bound at 0. Not a bug; do not "fix" it by adding the
  positive back into the partition function.
* Sinkhorn must run in the log domain with separate row/column updates: the graphs have
  `Nt=32 ≠ Nv=16` nodes, so any transpose-based update is dimensionally invalid.

## What still blocks a real reproduction

1. **The benchmarks.** MediaEval-2015 Twitter, Pheme and Weibo16 are not bundled in either
   repo. Without them Tables 1/3 cannot be produced.
2. **A ≥24 GB GPU (or an API key)** for the paper's actual VRP stage; `VRF_t`/`VRF_v` from
   `mock` are templates, not LLM reasoning, so the VRF ablation (`use_vrf`) is not a
   meaningful comparison yet.
3. **A trained SGG model** to replace the geometric predicate stand-in.
4. `en_core_web_trf` is installed but the smoke test used `en_core_web_sm` for speed; the
   paper's dependency parses come from the transformer pipeline.
5. Multi-seed runs (the paper reports mean over repeated runs) — `--seed` is wired through,
   and `ablation_table.py --seeds 3759,1,2` aggregates mean±sd per row.
