"""Sec.3.1.1 - Multimodal Entity Graph Construction (textual side).

G_t = (V_T, E_T): entities extracted from the post by dependency parsing, edges from the
syntactic relations between them, plus a global node v_g for the whole post and the four
generic type nodes (misc / person / location / organization) used by CMGAF (Sec.3.2.2).
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# relation vocabulary; index 0 is reserved for padding, 1 for self-loops
TEXT_RELATIONS = [
    '<pad>', '<self>', 'root', 'nsubj', 'nsubjpass', 'dobj', 'obj', 'iobj', 'attr',
    'appos', 'compound', 'nummod', 'poss', 'amod', 'advmod', 'agent', 'prep',
    'pobj', 'ccomp', 'xcomp', 'acl', 'relcl', 'conj', 'aux', 'auxpass', 'neg',
    'det', 'mark', 'punct', 'dep', 'nmod', 'case', 'cc', 'expl', 'global', 'type',
]
TEXT_REL2ID = {r: i for i, r in enumerate(TEXT_RELATIONS)}

GENERIC_TYPES = ['misc', 'person', 'location', 'organization']
GENERIC_TYPE_ID = {t: i for i, t in enumerate(GENERIC_TYPES)}

_NER_MAP = {
    'PERSON': 'person', 'PER': 'person',
    'GPE': 'location', 'LOC': 'location', 'FAC': 'location',
    'ORG': 'organization', 'NORP': 'organization',
}


@dataclass
class EntityGraph:
    nodes: List[str] = field(default_factory=list)          # node surface text
    types: List[str] = field(default_factory=list)          # generic type per node
    edges: List[Tuple[int, int, int]] = field(default_factory=list)   # (i, j, rel_id)
    global_index: int = -1
    generic_slice: Tuple[int, int] = (-1, -1)               # [start, end) of the 4 type nodes

    def add(self, text: str, type_: str = 'misc') -> int:
        self.nodes.append(text.strip() or 'unknown')
        self.types.append(type_)
        return len(self.nodes) - 1

    def add_edge(self, i: int, j: int, rel: str):
        self.edges.append((i, j, TEXT_REL2ID.get(rel, TEXT_REL2ID['dep'])))

    def adjacency(self, max_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (A [N,N] float32, R [N,N] int) padded to exactly N = max_nodes.

        The cache stores fixed-shape arrays and pads at build time, so every sample
        must come out at max_nodes regardless of how many entities it has. Padded rows
        keep A == 0 and relation id 0 ('<pad>'), and are zeroed again by the node mask.
        """
        n = min(len(self.nodes), max_nodes)
        A = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        R = np.zeros((max_nodes, max_nodes), dtype=np.int64)
        for i, j, r in self.edges:
            if i < n and j < n:
                A[i, j] = 1.0
                A[j, i] = 1.0          # dependency edges are treated as undirected
                R[i, j] = r
                R[j, i] = r
        for k in range(n):             # self-loops only for real nodes
            A[k, k] = 1.0
            R[k, k] = TEXT_REL2ID['<self>']
        return A, R


class TextGraphBuilder:
    def __init__(self, model_name='en_core_web_trf', max_nodes=32, device=None, lang='en'):
        self.lang = lang
        self.max_nodes = max_nodes
        self.model = self._load(model_name, lang)
        # spaCy only implements the English-style noun-chunk iterator for some languages;
        # Chinese has no word segmentation, so doc.noun_chunks raises ValueError there.
        self.noun_chunks_ok = self._supports_noun_chunks()

    @staticmethod
    def _load(model_name, lang='en'):
        """Load a pipeline *of the requested language*.

        Falling back across languages (English pipeline for Chinese text) used to be
        silent and produced meaningless entity spans, so the fallback below is only
        ever within the same language: trf -> sm.
        """
        import spacy
        from .lang import SPACY_BY_LANG
        alternatives = list(SPACY_BY_LANG.get(lang, ()))
        prefix = (model_name or '').split('_')[0]
        if len(prefix) == 2 and prefix != lang:
            raise OSError(f'spaCy pipeline "{model_name}" is a {prefix.upper()} model, '
                          f'but lang="{lang}" was requested. Alternatives: {alternatives}')
        # a typo'd name must not quietly become the language default: only continue to
        # the fallback chain when the requested pipeline is a recognised (or installed)
        # package that is simply missing from disk
        if model_name not in alternatives and not spacy.util.is_package(model_name):
            raise ValueError(
                f'unknown spaCy pipeline "{model_name}" for lang="{lang}". '
                f'Recognised options: {alternatives}, or omit the argument to use '
                f'the default')
        candidates = [model_name] + alternatives
        seen = []
        for name in candidates:
            if name in seen:
                continue
            seen.append(name)
            try:
                nlp = spacy.load(name, disable=['textcat'])
                if name != model_name:
                    print(f'[text_graph] "{model_name}" not installed, using "{name}"')
                return nlp
            except OSError:
                continue
        raise OSError(
            f'No usable spaCy pipeline for lang="{lang}" (tried {seen}). Install one:\n'
            f'    python -m spacy download {alternatives[0] if alternatives else model_name}')

    def _supports_noun_chunks(self) -> bool:
        try:
            _ = list(self.model('a dog on the mat').noun_chunks)
            return True
        except ValueError:
            # spaCy: "This method is not implemented for language ... without rules"
            return False
        except Exception:                                    # noqa: BLE001
            return False

    def build(self, text: str, with_generic_nodes: bool = True) -> EntityGraph:
        g = EntityGraph()
        doc = self.model(text[:4000])

        # 1) entity nodes from NER, merged over overlapping spans
        spans: List[Tuple[int, int, str, str]] = []
        for ent in doc.ents:
            spans.append((ent.start, ent.end, ent.text, _NER_MAP.get(ent.label_, 'misc')))
        for start, end, txt in self._candidate_phrases(doc):
            if not any(start >= s and end <= e for s, e, _, _ in spans):
                spans.append((start, end, txt, 'misc'))
        spans = self._dedupe(spans)

        token2node: Dict[int, int] = {}
        for start, end, s, t in spans:
            if len(g.nodes) >= self.max_nodes - 5:            # keep room for global + types
                break
            idx = g.add(s, t)
            for k in range(start, end):
                token2node[k] = idx

        # 2) global node v_g summarising the whole post
        g.global_index = g.add(self._headline(text), 'misc')
        g.add_edge(g.global_index, g.global_index, '<self>')

        # 3) edges from syntactic heads of the entity spans
        for (start, end, _, _), (s2, e2, _, _) in zip(spans, spans):
            if start == s2:
                continue
            i, j = token2node.get(start), token2node.get(s2)
            if i is None or j is None or i == j:
                continue
            rel = self._relation(doc, start, s2)
            g.add_edge(i, j, rel)
        for i in range(len(g.nodes)):
            if i != g.global_index:
                g.add_edge(i, g.global_index, 'global')

        if with_generic_nodes:
            self.append_generic_nodes(g)
        return g

    def _candidate_phrases(self, doc):
        """Event/object phrases that become 'misc' entity nodes.

        spaCy only implements its English-style noun-chunk rules for a subset of
        languages, and Weibo16 is Chinese, where `doc.noun_chunks` raises ValueError.
        Falling back to *nothing* would have emptied G_t of every object entity (which is
        exactly what the visual graph is aligned against), so for those languages maximal
        runs of nominal tokens are used instead.
        """
        if self.noun_chunks_ok:
            return [(c.start, c.end, c.text) for c in doc.noun_chunks]
        from .lang import is_cjk
        spans, cur = [], None
        for tok in doc:
            nominal = tok.pos_ in ('NOUN', 'PROPN', 'NUM') and not tok.is_punct
            if nominal:
                if cur is None:
                    cur = [tok.i, tok.i + 1, tok.text]
                else:
                    cur[1] = tok.i + 1
                    # Chinese is character-tokenised, so joining without spaces keeps the
                    # span readable and matchable; Latin scripts need the space back
                    cur[2] += tok.text if is_cjk(cur[2]) or is_cjk(tok.text) else ' ' + tok.text
            elif cur is not None:
                spans.append((cur[0], cur[1], cur[2]))
                cur = None
        if cur is not None:
            spans.append((cur[0], cur[1], cur[2]))
        return spans

    @staticmethod
    def _dedupe(spans):
        out = []
        for sp in sorted(spans, key=lambda x: (x[0], -(x[1] - x[0]))):
            if any(sp[0] >= k[0] and sp[1] <= k[1] for k in out):
                continue
            out.append(sp)
        return out

    @staticmethod
    def _headline(text: str, limit=128) -> str:
        t = re.sub(r'http\S+|@\w+|#\w+', ' ', text)
        t = re.sub(r'\s+', ' ', t).strip()
        return t[:limit] if t else 'no textual content'

    @staticmethod
    def _relation(doc, a: int, b: int) -> str:
        """Nearest dependency path label between token spans starting at a and b."""
        for hop in range(1, 5):
            for tok_idx in range(a, min(a + hop, len(doc))):
                tok = doc[tok_idx]
                head = tok.head.i
                if head == b:
                    return tok.dep_
                if head <= b and any(t.i == b for t in tok.ancestors):
                    child = [t for t in tok.ancestors if t.i == b][0]
                    return child.dep_
            for tok_idx in range(b, min(b + hop, len(doc))):
                tok = doc[tok_idx]
                if tok.head.i == a:
                    return tok.dep_
        return 'dep'

    def append_generic_nodes(self, g: EntityGraph):
        """Sec.3.2.2: four generic type nodes that bridge cross-modal entity nodes."""
        start = len(g.nodes)
        members: List[List[int]] = [[] for _ in GENERIC_TYPES]
        for i, t in enumerate(g.types):
            if i < start:
                members[GENERIC_TYPE_ID.get(t, 0)].append(i)
        for gi, name in enumerate(GENERIC_TYPES):
            idx = g.add(name, name)
            for m in members[gi]:
                g.add_edge(idx, m, 'type')
            if not members[gi]:
                g.add_edge(idx, g.global_index, 'type')
        g.generic_slice = (start, len(g.nodes))
        return g
