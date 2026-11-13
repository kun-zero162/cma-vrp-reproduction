"""Regression checks for language handling (Weibo16 / Chinese support).

Run from CMAVRP:  python tools/unit_lang.py

These exist because the Chinese path used to fail *silently*: the entity matcher deleted
every CJK character, and a missing Chinese spaCy pipeline fell back to the English one
without saying so. A cache built that way looks perfectly valid.
"""
import io
import os
import sys
import contextlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # CMAVRP/

# The Windows console defaults to cp1252 and cannot print CJK, which would crash the
# very checks that report Chinese entity names.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

import numpy as np  # noqa: E402

from preprocess.entity_matching import EntityMatcher, normalize  # noqa: E402
from preprocess.lang import (detect_lang, require_lang_support,  # noqa: E402
                             resolve_lang, cjk_ratio)
from preprocess.text_graph import TextGraphBuilder, EntityGraph  # noqa: E402
from preprocess.visual_graph import COCO_ZH, VisualGraphBuilder  # noqa: E402

TRAINING_ONLY_NOTE = 'needs the zh assets (spacy download zh_core_web_sm)'


def ok(msg):
    print(f'  [ok] {msg}')


def check_normalize_and_scoring():
    m = EntityMatcher(mode='string', threshold=0.6)
    zh_a, zh_b = normalize('这张图片中的火车站'), normalize('火车站')
    assert detect_lang('这张图片中的火车站') == 'zh' and detect_lang('a tweet') == 'en'
    assert cjk_ratio('火车站火车') == 1.0
    assert 0.3 <= cjk_ratio('火车站 train') < 0.5, 'mixed post: 3 CJK of 8 non-space chars'
    assert detect_lang('火车站 train') == 'zh', 'a code-switched post still routes to zh'
    assert zh_a != zh_b and '火' in zh_a, (zh_a, zh_b)   # CJK must survive normalise
    assert m._score(zh_a, zh_b, None, None) >= 0.6, 'chinese containment should match'
    # the containment guard its comment always promised: 'man' inside 'omnipresent'
    assert m._score(normalize('man'), normalize('omnipresent'), None, None) < 0.6
    assert m._score(normalize('new york'), normalize('New York City'), None, None) >= 0.6
    ok('CJK survives normalize(); containment guard rejects man/omnipresent')


def _graph(names, global_idx_offset=None):
    g = EntityGraph()
    for n in names:
        g.add(n, 'misc')
    g.global_index = g.add('headline summary text', 'misc')
    start = len(g.nodes)
    for t in ('misc', 'person', 'location', 'organization'):
        g.add(t, t)
    g.generic_slice = (start, len(g.nodes))
    return g


def check_chinese_matching_finds_real_links():
    """The actual Sec.3.2.1 mechanism, with real object names on both sides."""
    tg = _graph(['火车站', '消防栓', ' somebody'])
    vg = _graph(['火车', '时钟'])
    m = EntityMatcher(mode='string', threshold=0.6)
    A, pairs = m.match(tg, vg)
    real = [p for p in pairs if p[2] < 0.999]
    assert any(p[0] == 0 for p in real), f'火车站 should link to 火车, got {pairs}'
    ok(f'chinese textual entity matched visual entity ({len(real)} real link(s))')


def check_cross_lingual_warning():
    tg = _graph(['火车站', '消防栓'])
    vg = _graph(['train', 'clock'])            # untranslated detector labels
    m = EntityMatcher(mode='string', threshold=0.6)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        A, pairs = m.match(tg, vg)
    out = buf.getvalue()
    assert 'not localised' in out, out
    assert m.warned_no_entity_links
    assert all(p[2] == 1.0 for p in pairs), 'only the type bridges should remain'
    ok('untranslated visual labels are reported instead of looking like a real result')

    # the synthetic-data case actually seen here: grid proposals labelled 'unknown'
    m2 = EntityMatcher(mode='string', threshold=0.6)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m2.match(_graph(['person', 'the crowd']), _graph(['unknown', 'unknown']))
    assert 'grid proposal' in buf.getvalue(), buf.getvalue()
    ok('an all-"unknown" visual vocabulary is flagged as a detector problem')

    # and a healthy English match must NOT warn
    m3 = EntityMatcher(mode='string', threshold=0.6)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m3.match(_graph(['the fire hydrant', 'a clock']), _graph(['fire hydrant']))
    assert buf.getvalue() == '', repr(buf.getvalue())
    ok('matching graphs produce no warning (no false alarm)')


def check_guard_rejects_english_config_for_zh():
    """Each half of the misconfiguration must be caught on its own."""
    cases = {
        'en_core_web_sm': lambda m: 'zh_core_web' in m,          # wrong parser
        'zh_core_web_trf': lambda m: 'chinese' in m.lower() or '[UNK]' in m,
    }                                              # wrong encoder (and _trf absent)
    for model, expect in cases.items():
        try:
            require_lang_support('zh', model, 'google-bert/bert-base-uncased', strict=True)
        except RuntimeError as e:
            msg = str(e)
            assert expect(msg), f'{model}: unexpected guidance\n{msg}'
            assert '--allow_lang_mismatch' in msg, 'the escape hatch must be named'
            ok(f'guard rejects lang=zh with {model} ({len(msg)} chars of guidance)')
            continue
        raise AssertionError(f'guard accepted zh data with model={model}')


def test_guard_allows_correct_config():
    msg = require_lang_support('zh', 'zh_core_web_sm', 'bert-base-chinese', strict=True)
    assert msg == '', msg
    # and the English default must still pass
    assert require_lang_support('en', 'en_core_web_sm', 'bert-base-uncased') == ''
    ok('correct zh and en configurations pass the guard')


def test_no_cross_language_spacy_fallback():
    """A bad or mismatched pipeline name must be an error, never a silent substitution."""
    cases = [('nonexistent_pipeline_xyz', ValueError, 'unknown spaCy pipeline'),
             ('en_core_web_sm', OSError, 'EN model')]      # right lang must be enforced
    for model, exc, needle in cases:
        try:
            TextGraphBuilder(model, max_nodes=32, lang='zh')
        except (OSError, ValueError) as e:
            assert type(e) is exc, f'{model}: raised {type(e).__name__}, expected {exc.__name__}'
            assert needle.lower() in str(e).lower(), f'{model}: {e}'
            ok(f'{model} rejected ({type(e).__name__})')
            continue
        raise AssertionError(f'TextGraphBuilder accepted model={model} for lang=zh')
    # and the recognised convenience still works: a missing but recognised pipeline name
    # must fall back to the same-language alternative and SAY so (never silently)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        b = TextGraphBuilder('zh_core_web_trf', max_nodes=32, lang='zh')
    assert b.lang == 'zh', 'language attribute lost'
    assert 'not installed' in buf.getvalue(), (
        'zh_core_web_trf is absent here, so the fallback must be announced; got: '
        + repr(buf.getvalue()))
    ok('missing zh_core_web_trf falls back visibly to zh_core_web_sm')
    # same convenience on the English side, so the fix cannot strand en users
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        e = TextGraphBuilder('en_core_web_sm', max_nodes=32, lang='en')
    assert e.noun_chunks_ok is True, 'English must keep the noun-chunk path'
    assert 'not installed' not in buf.getvalue(), 'installed en model should not warn'
    ok('en pipeline loads without a spurious fallback notice')


def test_chinese_parser_produces_entity_nodes():
    try:
        b = TextGraphBuilder('zh_core_web_sm', max_nodes=32, lang='zh')
    except OSError as e:
        print(f'  [skip] {TRAINING_ONLY_NOTE}: {e}')
        return None
    assert b.noun_chunks_ok is False, 'spaCy should not support noun chunks for zh'
    g = b.build('震惊！这张图片证明时钟被隐藏在消防栓和人群的人群中！')
    ents = [n for i, n in enumerate(g.nodes) if i not in (g.global_index,)
            and not (g.generic_slice[0] <= i < g.generic_slice[1])]
    assert ents, 'Chinese parse produced no entity nodes (POS fallback failed)'
    assert any('消防' in e or '时钟' in e for e in ents), ents
    ok(f'zh pipeline + POS fallback yields entity nodes: {ents[:4]}')
    return b


def test_visual_surface_localisation_keeps_type_inference():
    vb = VisualGraphBuilder(detector='none', lang='zh')
    assert vb._surface('train') == COCO_ZH['train'] == '火车'
    assert vb._surface('not_a_coco_class') == 'not_a_coco_class'
    assert vb._type_of('train') == 'organization'      # still driven by the English label
    zh_global = vb._global_text('配文测试', [('train', 0.9, [0, 0, 1, 1])])
    assert '火车' in zh_global and '配文' in zh_global, zh_global
    en = VisualGraphBuilder(detector='none', lang='en')
    assert en._global_text('a caption', [('train', 0.9, [0, 0, 1, 1])]) == \
        'the image shows train; caption: a caption'
    assert en._global_text('', []) == 'the image shows unrecognised scene'
    ok('visual node + global-node text is Chinese while relation/type vocab stays English')


def test_zh_encoder_does_not_collapse_to_unk():
    """The failure this all protects: an English BERT maps every Chinese char to UNK."""
    if not os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'bert', 'bert-base-chinese')):
        print('  [skip] bert-base-chinese not downloaded')
        return
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from preprocess.build_dataset import BertNodeEncoder
    import torch
    enc = BertNodeEncoder('bert-base-chinese', 'cpu', batch_size=8)
    v = enc.encode(['火车', '时钟', '消防栓', 'empty'])
    sim = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    assert np.isfinite(v).all() and v.shape == (4, 768), v.shape
    assert sim(v[0], v[1]) < 0.995, 'distinct Chinese entities share one embedding (=UNK)'
    assert sim(v[0], v[2]) < 0.995
    ok(f'bert-base-chinese separates entities (cos 火车/时钟={sim(v[0], v[1]):.3f}, '
       f'火车/消防栓={sim(v[0], v[2]):.3f})')

    # and the English encoder must be *rejected* for Chinese text, quantitatively
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'bert', 'bert-base-uncased'))
    ids = t('火车站', add_special_tokens=False)['input_ids']
    unk = t.convert_ids_to_tokens(ids).count('[UNK]')
    assert unk >= 2, f'expected English BERT to UNK Chinese, got {ids}'
    ok(f'confirmed the hazard: bert-base-uncased turns 火车站 into {unk} [UNK] tokens')


def main():
    print('[1/8] normalisation and string scoring');   check_normalize_and_scoring()
    print('[2/8] chinese entity matching');            check_chinese_matching_finds_real_links()
    print('[3/8] cross-lingual warning');              check_cross_lingual_warning()
    print('[4/8] strict guard');                       check_guard_rejects_english_config_for_zh()
    print('[5/8] guard accepts correct config');       test_guard_allows_correct_config()
    print('[6/8] no cross-language spaCy fallback');   test_no_cross_language_spacy_fallback()
    print('[7/8] visual surface localisation')
    b = test_chinese_parser_produces_entity_nodes()
    test_visual_surface_localisation_keeps_type_inference()
    print('[8/8] node encoder vocabulary');            test_zh_encoder_does_not_collapse_to_unk()
    assert resolve_lang('weibo') == 'zh' and resolve_lang('twitter') == 'en'
    print('\nALL LANGUAGE CHECKS PASSED' if b else '\nALL CHECKS PASSED (zh parse skipped)')


if __name__ == '__main__':
    main()
