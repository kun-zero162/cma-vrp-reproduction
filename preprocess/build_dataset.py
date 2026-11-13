"""Preprocessing pipeline: raw posts + images -> cached multimodal entity graphs.

    data/{dataset}/{split}.json  ->  cache/{dataset}/{split}.npz (+ .ids.json, meta.json)

Raw record format (one dict per line-object in a JSON list):
    {"id": "...", "text": "post content", "image": "imgs/xxx.jpg", "label": 0|1}
label 0 = real, 1 = fake (Twitter / Pheme / Weibo all map onto this).

Run from the CMAVRP directory:
    python -m preprocess.build_dataset --datasets twitter --splits train val test
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # CMAVRP/
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Chinese node texts reach both stdout (warnings) and the .nodes.json dumps; the Windows
# console's cp1252 default would raise UnicodeEncodeError halfway through a cache build.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

from configs.config_setting import setting_config as config  # noqa: E402
from preprocess.entity_matching import EntityMatcher  # noqa: E402
from preprocess.lang import (DATASET_LANG, NODE_ENCODER_BY_LANG, SPACY_BY_LANG,  # noqa: E402
                             detect_lang, require_lang_support, resolve_lang)
from preprocess.text_graph import TextGraphBuilder  # noqa: E402
from preprocess.visual_graph import VisualGraphBuilder  # noqa: E402
from preprocess.vrp import VisualReasoningPrompter  # noqa: E402


class BertNodeEncoder:
    """[S4] node / reasoning texts are initialised with BERT-base-uncased."""

    def __init__(self, model_name, device, batch_size=64, max_len=64):
        from transformers import AutoModel, AutoTokenizer
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.batch_size = batch_size
        self.max_len = max_len

    @torch.no_grad()
    def encode(self, texts):
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        out = []
        for i in range(0, len(texts), self.batch_size):
            chunk = [t if t else 'empty' for t in texts[i:i + self.batch_size]]
            enc = self.tokenizer(chunk, padding='max_length', truncation=True,
                                 max_length=self.max_len, return_tensors='pt').to(self.device)
            hidden = self.model(**enc).last_hidden_state          # [B,L,768]
            mask = enc['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-6)   # mean pooling
            out.append(pooled.float().cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)


def load_records(dataset, split, limit=None):
    path = os.path.join(config.raw_dir, dataset, f'{split}.json')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'{path} not found. Expected a JSON list of '
            f'{{"id", "text", "image", "label"}} records.')
    with open(path, encoding='utf-8') as f:
        recs = json.load(f)
    return recs[:limit] if limit else recs


def resolve_image(dataset, rec):
    img = rec.get('image') or ''
    for cand in (os.path.join(config.raw_dir, dataset, img),
                 os.path.join(config.raw_dir, dataset, 'imgs', os.path.basename(img))):
        if os.path.isfile(cand):
            return cand
    return None


def empty_visual_graph(max_v):
    """Fallback when a post has no readable image: a degenerate graph with only the
    global node and the four generic type nodes, so the batch stays well-shaped."""
    from preprocess.text_graph import EntityGraph
    from preprocess.visual_graph import VISUAL_REL2ID
    vg = EntityGraph()
    vg.global_index = vg.add('no image attached', 'misc')
    start = len(vg.nodes)
    for name in ('misc', 'person', 'location', 'organization'):
        idx = vg.add(name, name)
        vg.add_edge(idx, vg.global_index, 'type')
    vg.generic_slice = (start, len(vg.nodes))
    vg.add_edge(vg.global_index, vg.global_index, '<self>')
    A, R = vg.adjacency(max_v)
    return vg, A, R, np.zeros((max_v, 2048), dtype=np.float32)


def build_split(dataset, split, builder, vbuilder, matcher, prompter, encoder,
                max_t, max_v, limit=None, verbose=True):
    from PIL import Image
    recs = load_records(dataset, split, limit)
    store = {k: [] for k in ('X_t', 'X_v', 'X_v_vis', 'A_t', 'A_v', 'A_tv',
                             'VRF_t', 'VRF_v', 'mask_t', 'mask_v', 'y', 'R_t', 'R_v')}
    ids, node_dump = [], []

    for ri, rec in enumerate(recs):
        text = rec.get('text') or rec.get('claim') or ''
        label = int(rec.get('label', 0))
        img_path = resolve_image(dataset, rec)

        tg = builder.build(text)
        if img_path is not None:
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:  # noqa: BLE001
                print(f'[warn] unreadable image {img_path}: {e}')
                img = Image.new('RGB', (224, 224), (128, 128, 128))
            vg, Av, Rv, vis, boxes, gf = vbuilder.build(img, img_path, caption=text)
        else:
            vg, Av, Rv, vis = empty_visual_graph(max_v)
        At, Rt = tg.adjacency(max_t)

        A_tv, pairs = matcher.match(tg, vg)
        # truth-driven prompting only ever uses the label of the *training* split; test
        # splits are prompted without ground truth, matching the paper's inference setting
        vrf_t_txt, vrf_v_txt = prompter.reason(
            text, tg.nodes[:max_t], vg.nodes[:max_v],
            label=label if split == 'train' else None, image_path=img_path)

        X_t = encoder.encode(tg.nodes[:max_t])
        X_v = encoder.encode(vg.nodes[:max_v])
        VRF_t = encoder.encode(list(vrf_t_txt))
        VRF_v = encoder.encode(list(vrf_v_txt))

        def pad_rows(a, n, dim):
            out = np.zeros((n, dim), dtype=np.float16)
            k = min(a.shape[0], n)
            out[:k] = a[:k].astype(np.float16)
            return out

        nt, nv = min(len(tg.nodes), max_t), min(len(vg.nodes), max_v)
        store['X_t'].append(pad_rows(X_t, max_t, 768))
        store['X_v'].append(pad_rows(X_v, max_v, 768))
        store['VRF_t'].append(pad_rows(VRF_t, max_t, 768))
        store['VRF_v'].append(pad_rows(VRF_v, max_v, 768))
        store['X_v_vis'].append(pad_rows(vis, max_v, vis.shape[-1]))
        store['A_t'].append(At)
        store['A_v'].append(Av)
        store['R_t'].append(Rt.astype(np.int8))
        store['R_v'].append(Rv.astype(np.int8))
        A_tv_full = np.zeros((max_t, max_v), dtype=np.float32)
        k = min(A_tv.shape[0], max_t)
        j = min(A_tv.shape[1], max_v)
        A_tv_full[:k, :j] = A_tv[:k, :j]
        store['A_tv'].append(A_tv_full)
        mt = np.zeros(max_t, dtype=np.int8); mt[:nt] = 1
        mv = np.zeros(max_v, dtype=np.int8); mv[:nv] = 1
        store['mask_t'].append(mt)
        store['mask_v'].append(mv)
        store['y'].append(np.int64(label))
        ids.append(str(rec.get('id', ri)))
        node_dump.append({'id': str(rec.get('id', ri)), 'text_nodes': tg.nodes[:max_t],
                          'visual_nodes': vg.nodes[:max_v], 'matched': len(pairs)})

        if verbose and (ri + 1) % 25 == 0:
            print(f'  [{dataset}/{split}] {ri + 1}/{len(recs)} '
                  f'(nt={nt}, nv={nv}, links={len(pairs)})', flush=True)

    os.makedirs(os.path.join(config.cache_path, dataset), exist_ok=True)
    out_npz = os.path.join(config.cache_path, dataset, f'{split}.npz')
    # np.stack's own error does not say which array is ragged, so report it here
    for k, v in store.items():
        shapes = {a.shape for a in v}
        if len(shapes) > 1:
            raise AssertionError(f'ragged cache array {k} for {dataset}/{split}: '
                                 f'{sorted(shapes)}')
    np.savez_compressed(out_npz, **{k: np.stack(v) for k, v in store.items()})
    with open(os.path.join(config.cache_path, dataset, f'{split}.ids.json'), 'w',
              encoding='utf-8') as f:
        json.dump(ids, f)
    with open(os.path.join(config.cache_path, dataset, f'{split}.nodes.json'), 'w',
              encoding='utf-8') as f:
        json.dump(node_dump, f, ensure_ascii=False)
    n_links = sum(d['matched'] for d in node_dump)
    print(f'[ok] {out_npz}: {len(ids)} samples, {n_links} cross-modal links '
          f'({n_links / max(1, len(ids)):.1f} per sample)')
    return len(ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', default=config.datasets, nargs='+',
                   choices=['twitter', 'pheme', 'weibo'])
    p.add_argument('--splits', default=['train', 'val', 'test'], nargs='+')
    p.add_argument('--limit', type=int, default=None, help='cap samples per split (smoke test)')
    p.add_argument('--device', default=config.device, choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--vrp_backend', default=config.vrp_backend, choices=['mock', 'hf', 'agent'])
    p.add_argument('--batch_size', type=int, default=64, help='BERT encoding batch size')
    p.add_argument('--spacy_model', default=None,
                   help='override the pipeline name; defaults per --lang '
                        '(en_core_web_trf / zh_core_web_sm)')
    p.add_argument('--lang', default=config.lang, choices=['auto', 'en', 'zh'],
                   help="'auto' takes the language from the dataset name "
                        f"({DATASET_LANG}) and confirms it against the posts")
    p.add_argument('--allow_lang_mismatch', action='store_true',
                   help='build anyway when the parser/encoder cannot handle the '
                        "dataset's language; the cache will be degraded")
    p.add_argument('--detector', default=config.detector,
                   help='DETR checkpoint, or "none" to skip object detection')
    args = p.parse_args()

    config.cache_path = os.path.join(_HERE, 'cache')
    device = ('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else
              ('cpu' if args.device == 'auto' else args.device))

    vbuilder = VisualGraphBuilder(
        detector=args.detector, ocr_backend=config.ocr_backend,
        top_k=config.top_k_objects, max_nodes=config.max_visual_nodes,
        use_region_feats=config.use_region_feats, device=device)
    # only the entity *surface language* differs, so one DETR instance is re-targeted per
    # dataset instead of loading a second copy of the detector (a real cost on 4 GB)
    prompter = VisualReasoningPrompter(backend=args.vrp_backend,
                                       llm_text=config.llm_text,
                                       lvlm_visual=config.lvlm_visual,
                                       device=device)

    # The parser, the node encoder and the matcher are all language-specific, so they are
    # built per dataset: a single shared builder would force one language on a mixed
    # `--datasets twitter weibo` run.
    components = {}

    def get_components(ds):
        sample = ''
        try:
            recs = load_records(ds, args.splits[0], limit=1)
            sample = (recs[0].get('text') or recs[0].get('claim') or '') if recs else ''
        except (FileNotFoundError, OSError):
            pass
        lang = resolve_lang(ds, args.lang, sample)
        if sample and detect_lang(sample) != lang:
            print(f'[warn] {ds}: configured lang="{lang}" but the first post reads as '
                  f'"{detect_lang(sample)}" -- check the raw data or pass --lang '
                  f'deliberately.')
        if lang in components:
            return lang, components[lang]

        spacy_model = args.spacy_model or {
            'en': config.spacy_model, 'zh': config.spacy_model_zh}.get(lang, args.spacy_model)
        bert = {"en": config.bert_path,
                "zh": config.bert_path_zh}.get(lang, config.bert_path)
        msg = require_lang_support(lang, spacy_model, bert,
                                  strict=not args.allow_lang_mismatch)
        if msg:
            print('[warn] ' + msg.replace('\n', '\n  '))
        print(f'# dataset={ds} lang={lang} spacy={spacy_model} bert={bert} '
              f'vrp={args.vrp_backend} device={device} #')
        components[lang] = (
            TextGraphBuilder(spacy_model, max_nodes=config.max_text_nodes, lang=lang),
            EntityMatcher(mode='string', threshold=0.6),
            BertNodeEncoder(bert, device, batch_size=args.batch_size))
        return lang, components[lang]

    total = {}
    for ds in args.datasets:
        lang, (builder, matcher, encoder) = get_components(ds)
        vbuilder.lang = lang
        total[ds] = {'lang': lang}
        for split in args.splits:
            try:
                total[ds][split] = build_split(
                    ds, split, builder, vbuilder, matcher, prompter, encoder,
                    config.max_text_nodes, config.max_visual_nodes, args.limit)
            except FileNotFoundError as e:
                print(f'[skip] {e}')
    with open(os.path.join(config.cache_path, args.datasets[0], 'build_meta.json'),
              'w', encoding='utf-8') as f:
        json.dump({'device': device, 'vrp_backend': args.vrp_backend,
                   'datasets': total,
                   'max_text_nodes': config.max_text_nodes,
                   'max_visual_nodes': config.max_visual_nodes,
                   'supported_langs': sorted(SPACY_BY_LANG),
                   'node_encoders': NODE_ENCODER_BY_LANG}, f, indent=2)


if __name__ == '__main__':
    main()
