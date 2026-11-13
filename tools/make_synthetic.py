"""Synthetic Twitter/Pheme/Weibo-format data so the pipeline can be exercised end to end.

The real benchmarks (MediaEval Twitter, Pheme, Weibo16) are not shipped with either repo
and must be downloaded separately. Until they are in place this generator writes the same
raw layout that `preprocess/build_dataset.py` expects:

    data/{dataset}/{train,val,test}.json  +  data/{dataset}/imgs/*.jpg

Each post gets a rendered scene image and a caption; for label=1 (fake) the caption
contradicts the objects actually drawn, which is the signal the model is meant to learn
from, so the smoke test exercises real cross-modal misalignment rather than noise.
"""
import argparse
import json
import os
import random
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # CMAVRP/
from preprocess.visual_graph import COCO_ZH  # noqa: E402

# COCO class names that DETR reliably detects on simple renders
COCO = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'bench', 'tv', 'laptop',
        'chair', 'dining table', 'potted plant', 'bed', 'clock', 'tv monitor']

TEMPLATES_REAL = [
    'Breaking: photos show {a} and {b} at the scene. Authorities confirm the report.',
    'Witnesses posted this image of a {a} next to a {b}. Verified on the ground.',
    'The {a} and the {b} were photographed together during the event today.',
    'Local news: {a} near the {b} as the situation develops. Image is genuine.',
]
TEMPLATES_FAKE = [
    'SHOCKING: this image proves {c} was hidden among the crowd of {a} and {b}!',
    'Viral post claims a {c} attacked people beside the {a}. No source given.',
    'They say this photo shows a {c} at the protest with the {a} and {b}.',
    'URGENT: {c} caught on camera destroying the {a} while the {b} watched.',
]

# Weibo16 is Chinese [S2]; these mirror the templates above so the Chinese plumbing is
# exercised with the same real/fake contrast (fake captions name an absent object).
TEMPLATES_REAL_ZH = [
    '\u73b0\u573a\u62cd\u7167\u663e\u793a {a} \u548c {b} \u51fa\u73b0\u5728\u4e8b\u53d1\u5730\uff0c\u6d88\u606f\u5df2\u5f97\u5230\u8bc1\u5b9e\u3002',
    '\u76ee\u51fb\u8005\u53d1\u5e03\u4e86\u8fd9\u5f20 {a} \u4e0e {b} \u76f8\u9694\u7684\u56fe\u7247\uff0c\u6765\u6e90\u53ef\u9760\u3002',
    '\u4eca\u65e5\u4e8b\u4ef6\u4e2d\uff0c{a} \u4e0e {b} \u88ab\u4e00\u540c\u62cd\u5230\uff0c\u56fe\u7247\u5c5e\u5b9e\u3002',
    '\u672c\u5730\u5a92\u4f53\u62a5\u9053\uff1a{a} \u5728 {b} \u9644\u8fd1\u51fa\u73b0\uff0c\u73b0\u573a\u56fe\u7247\u771f\u5b9e\u3002',
]
TEMPLATES_FAKE_ZH = [
    '\u9707\u60ca\uff01\u8fd9\u5f20\u56fe\u7247\u8bc1\u660e {c} \u88ab\u9690\u85cf\u5728 {a} \u548c {b} \u7684\u4eba\u7fa4\u4e2d\uff01',
    '\u7f51\u4f20\u89c6\u9891\u79f0 {c} \u88ad\u51fb\u4e86 {a} \u65c1\u8fb9\u7684\u4eba\uff0c\u6ca1\u6709\u6d88\u606f\u6765\u6e90\u3002',
    '\u4ed6\u4eec\u8bf4\u8fd9\u5f20\u7167\u7247\u663e\u793a\u6297\u8bae\u73b0\u573a\u6709 {c}\uff0c\u65c1\u8fb9\u662f {a} \u548c {b}\u3002',
    '\u7d27\u6025\uff1a{c} \u88ab\u62cd\u5230\u6b63\u5728\u7834\u574f {a}\uff0c\u800c {b} \u5728\u65c1\u8fb9\u770b\u7740\u3002',
]


def draw_scene(path, objects, rng):
    W = H = 320
    img = Image.new('RGB', (W, H), (int(rng.random() * 120) + 100,
                                    int(rng.random() * 120) + 100,
                                    int(rng.random() * 120) + 120))
    d = ImageDraw.Draw(img)
    boxes = []
    for obj in objects:
        w = rng.randint(60, 130)
        h = rng.randint(60, 130)
        x = rng.randint(0, W - w)
        y = rng.randint(0, H - h)
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        if rng.random() < 0.5:
            d.ellipse([x, y, x + w, y + h], fill=color, outline=(0, 0, 0), width=3)
        else:
            d.rectangle([x, y, x + w, y + h], fill=color, outline=(0, 0, 0), width=3)
        # the label burned into the image stays English: Pillow's default bitmap font has
        # no CJK glyphs, and OCR is disabled anyway (ocr_backend='none')
        d.text((x + 4, y + 4), obj[:12], fill=(255, 255, 255) if sum(color) < 380 else (0, 0, 0))
        boxes.append((obj, (x, y, x + w, y + h)))
    img.save(path, 'JPEG', quality=92)
    return boxes


def make(out_dir, dataset, counts, seed, lang='en'):
    rng = random.Random(seed)
    img_dir = os.path.join(out_dir, dataset, 'imgs')
    os.makedirs(img_dir, exist_ok=True)
    vocab = [c for c in COCO]
    real_t, fake_t = (TEMPLATES_REAL_ZH, TEMPLATES_FAKE_ZH) if lang == 'zh' \
        else (TEMPLATES_REAL, TEMPLATES_FAKE)
    name = (lambda x: COCO_ZH.get(x, x)) if lang == 'zh' else (lambda x: x)
    for split, n in counts.items():
        recs = []
        for i in range(n):
            present = rng.sample(vocab, k=rng.randint(2, 3))
            fake = rng.random() < 0.5
            if fake:
                absent = rng.choice([v for v in vocab if v not in present])
                text = rng.choice(fake_t).format(a=name(present[0]), b=name(present[-1]),
                                                 c=name(absent))
            else:
                text = rng.choice(real_t).format(a=name(present[0]), b=name(present[-1]))
            fname = f'{split}_{i:04d}.jpg'
            draw_scene(os.path.join(img_dir, fname), present, rng)
            recs.append({'id': f'{dataset}-{split}-{i}', 'text': text,
                         'image': f'imgs/{fname}', 'label': 1 if fake else 0,
                         'objects': present})
        with open(os.path.join(out_dir, dataset, f'{split}.json'), 'w', encoding='utf-8') as f:
            json.dump(recs, f, indent=1, ensure_ascii=False)
        n_fake = sum(r['label'] for r in recs)
        print(f'  {dataset}/{split}: {len(recs)} posts lang={lang} ({n_fake} fake, '
              f'{len(recs) - n_fake} real)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['twitter'],
                   choices=['twitter', 'pheme', 'weibo'])
    p.add_argument('--n_train', type=int, default=80)
    p.add_argument('--n_val', type=int, default=24)
    p.add_argument('--n_test', type=int, default=24)
    p.add_argument('--seed', type=int, default=3759)
    p.add_argument('--lang', default='auto', choices=['auto', 'en', 'zh'],
                   help="'auto' uses the benchmark's language (weibo -> zh)")
    p.add_argument('--out', default=None)
    args = p.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # CMAVRP/
    out_dir = args.out or os.path.join(here, 'data')
    counts = {'train': args.n_train, 'val': args.n_val, 'test': args.n_test}
    for ds in args.datasets:
        lang = args.lang if args.lang != 'auto' else ('zh' if ds == 'weibo' else 'en')
        print(f'# {ds} -> {os.path.join(out_dir, ds)} (lang={lang})')
        make(out_dir, ds, counts, args.seed, lang)


if __name__ == '__main__':
    main()
