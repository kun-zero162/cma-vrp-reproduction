#!/bin/bash
# CMA-VRP (ACM MM'25, DOI 10.1145/3746027.3754591) - full reproduction run.
# Run from CMAVRP/. Cache must already exist (see preprocess/build_dataset.py).
#
# Sec.4 protocol: lr 1e-3, StepLR decay 0.5 every 5 epochs, early stop 5/50,
# lambda=0.5 on L_cl, beta=0.1 on L_adv, seed 3759.
#
# The paper's VRP stage uses Mistral-7B + LLaVA-1.6-13B locally, which needs >=24 GB
# VRAM. On a smaller card use `agent` (API) or `mock`; the graph cache is built once by
# build_dataset.py, so the training loop itself only needs ~4 GB.

set -e

VRP=${VRP:-agent}                 # mock | agent | hf
GPU=${GPU:-0}
DATASETS=${DATASETS:-"twitter pheme weibo"}

# weibo is Chinese: its cache must be built with a zh pipeline and bert-base-chinese,
# otherwise preprocess/lang.py raises instead of silently parsing Chinese with English
# models. Build it once with
#   python -m spacy download zh_core_web_sm
#   python -m preprocess.build_dataset --datasets weibo --splits train val test \
#          --lang zh --spacy_model zh_core_web_sm

for DATASET in ${DATASETS}; do
    echo "===== ${DATASET} (vrp=${VRP}) ====="
    python train.py \
        --datasets "${DATASET}" \
        --vrp_backend "${VRP}" \
        --batch_size 32 \
        --epochs 50 \
        --early_stop 5 \
        --lr 1e-3 \
        --seed 3759 \
        --gpu "${GPU}"

    # test.py / robustness.py pick cuda:0 unless --cpu is given; only train.py takes --gpu
    python test.py --datasets "${DATASET}" --vrp_backend "${VRP}"
    python robustness.py --datasets "${DATASET}" --vrp_backend "${VRP}"
done

# ---- Table 2: each component removed in turn, on every dataset ----
# tools/ablation_table.py runs these and aggregates mean/sd + delta vs `full` into
# results/ablation_<dataset>.{md,csv,json}; the loop below is the same thing raw.
for DATASET in ${DATASETS}; do
  for ABL in use_vrf use_gcl_inter use_gcl_intra use_cmgaf use_accil use_vsca; do
    echo "===== ${DATASET} ablation: -${ABL} ====="
    python train.py \
        --datasets "${DATASET}" --vrp_backend "${VRP}" --gpu "${GPU}" \
        --batch_size 32 --epochs 50 --early_stop 5 --lr 1e-3 --seed 3759 \
        --ablate "${ABL}" --tag "no_${ABL}"
    python test.py --datasets "${DATASET}" --vrp_backend "${VRP}" \
        --ablate "${ABL}" --tag "no_${ABL}"
  done
done

# ---- CPU smoke test on the synthetic stand-in dataset (no GPU needed) ----
# python tools/make_synthetic.py --datasets twitter --n_train 60 --n_val 16 --n_test 16
# python -m preprocess.build_dataset --datasets twitter --splits train val test \
#        --device cpu --spacy_model en_core_web_sm --vrp_backend mock
# python train.py --datasets twitter --cpu --epochs 12 --early_stop 5 --batch_size 8
