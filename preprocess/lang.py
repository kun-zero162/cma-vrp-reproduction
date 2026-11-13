"""Language handling for the entity-graph pipeline.

Why this module exists
----------------------
The paper [S4] initialises every node with BERT-base-uncased and parses posts with
spaCy `en_core_web_trf`, i.e. it describes an English pipeline -- yet one of its three
benchmarks, Weibo16, is Chinese. Nothing in the original code made that dependency
explicit, so `--datasets weibo` used to "work" while producing garbage:

  * the English BERT vocabulary has no Chinese characters, so node texts became [UNK];
  * `en_core_web_*` cannot segment Chinese, so entity spans were essentially random;
  * `entity_matching.normalize()` stripped every non-ASCII character, collapsing all
    Chinese entities to a single space -- which then compared *equal* to each other.

`detect_lang()` and `require_lang_support()` turn those silent failures into either a
correct Chinese configuration or a hard error naming the packages to install.
"""
import re
from typing import Dict, Tuple

# [S4] the paper's benchmarks, with the language each one is written in
DATASET_LANG: Dict[str, str] = {'twitter': 'en', 'pheme': 'en', 'weibo': 'zh'}

# requested model, then the CPU-friendly fallback of the *same* language
SPACY_BY_LANG: Dict[str, Tuple[str, str]] = {
    'en': ('en_core_web_trf', 'en_core_web_sm'),
    'zh': ('zh_core_web_trf', 'zh_core_web_sm'),
}

# node-initialisation encoders. bert-base-chinese is the standard Chinese counterpart
# of bert-base-uncased: same 768 hidden size, so model_config['node_dim'] still holds.
NODE_ENCODER_BY_LANG: Dict[str, str] = {
    'en': 'google-bert/bert-base-uncased',
    'zh': 'google-bert/bert-base-chinese',
}

_CJK = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')


def cjk_ratio(text: str) -> float:
    """Fraction of non-space characters that are CJK ideographs."""
    chars = [c for c in (text or '') if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if _CJK.match(c)) / len(chars)


def detect_lang(text: str, threshold: float = 0.3) -> str:
    """'zh' when the sample is mostly Chinese characters, else 'en'.

    Only used to *confirm* the dataset's expected language, never to override it: a
    mixed-language post must not silently switch the whole run to another encoder.
    """
    return 'zh' if cjk_ratio(text) >= threshold else 'en'


def resolve_lang(dataset: str, lang: str = 'auto', sample_text: str = '') -> str:
    """Map a dataset name (or 'auto') to a language code."""
    if lang != 'auto':
        return lang
    if dataset in DATASET_LANG:
        return DATASET_LANG[dataset]
    return detect_lang(sample_text)


def is_cjk(s: str) -> bool:
    return bool(_CJK.search(s or ''))


def space_cjk(s: str) -> str:
    """Insert spaces around every CJK character so that whitespace tokenisation,
    `len(x.split())` and SequenceMatcher ratios behave like character-level matching.

    Chinese has no word separators, so the English string heuristics in
    EntityMatcher._score would otherwise compare unsegmented runs of characters.
    """
    return _CJK.sub(lambda m: ' ' + m.group(0) + ' ', s or '')


def require_lang_support(lang: str, spacy_model: str, bert_name: str,
                        strict: bool = True) -> str:
    """Check that the configured parser and encoder can actually handle `lang`.

    Returns a warning string when the configuration is usable but not ideal; raises
    RuntimeError (when strict) with copy-pasteable install commands otherwise.
    """
    import spacy

    problems = []

    wanted = SPACY_BY_LANG.get(lang)
    if wanted is None:
        problems.append(f'no spaCy pipeline is configured for language "{lang}"; '
                        f'supported: {sorted(SPACY_BY_LANG)}')
    else:
        # the *requested* pipeline must belong to this language: checking only whether
        # some zh pipeline happens to be installed would let `--spacy_model
        # en_core_web_sm` through and rebuild the silent-garbage path
        prefix = (spacy_model or '').split('_')[0]
        if len(prefix) == 2 and prefix != lang:
            problems.append(
                f'spaCy pipeline "{spacy_model}" parses {prefix.upper()} text but the '
                f'data is {lang.upper()}; use one of {list(wanted)} (or leave '
                f'--spacy_model empty to pick it automatically)')
        elif not any(spacy.util.is_package(m) for m in wanted):
            problems.append(
                'no {lang} spaCy pipeline installed (looked for {a} / {b}). Run one of:'
                .format(lang=lang.upper(), a=wanted[0], b=wanted[1]))
            problems.extend(f'    python -m spacy download {m}' for m in wanted)

    # An English BERT has no CJK in its vocabulary: Chinese node texts would encode as
    # [UNK] and every entity would share one embedding.
    expect = NODE_ENCODER_BY_LANG.get(lang, '')
    if lang == 'zh' and 'chinese' not in bert_name.lower() and 'mBERT'.lower() not in bert_name:
        problems.append(
            f'node encoder "{bert_name}" cannot encode Chinese (every character becomes '
            f'[UNK]); expected one of: {expect}, or a multilingual encoder such as '
            f'google-bert/bert-base-multilingual-cased')
    if lang == 'en' and ('chinese' in bert_name.lower()):
        problems.append(f'node encoder "{bert_name}" is Chinese but the data is English')

    if not problems:
        return ''
    msg = ('language/configuration mismatch for lang=%s dataset support:\n  - %s'
           % (lang, '\n  - '.join(problems)))
    if strict:
        raise RuntimeError(msg + '\n[hint] pass --allow_lang_mismatch to proceed anyway '
                                'and accept degraded entity graphs (not recommended: the '
                                'resulting cache is unusable for reporting).')
    return msg
