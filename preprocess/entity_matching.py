"""Sec.3.2.1 (i) Entity Matching: links textual entities v_Ti to visual entities v_Ij.

The matched pairs are exactly the supervision used by the inter-graph contrastive loss
(A_tv is also the cross adjacency consumed by CMGAF in Sec.3.2.2).
"""
import difflib
import re

import numpy as np

from .lang import is_cjk, space_cjk
from .text_graph import GENERIC_TYPES
from .visual_graph import VISUAL_REL2ID

# Unicode-aware: the previous [^a-z0-9 ]+ pattern deleted every CJK character, so all
# Chinese entities normalised to a lone space -- which then compared equal to every other
# collapsed entity and produced perfect but meaningless links (and zero real ones).
_NONWORD = re.compile(r'[^\w\s]+', re.UNICODE)


def normalize(s: str) -> str:
    s = (s or '').lower().strip()
    s = _NONWORD.sub(' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    # Chinese has no word separators; spacing the ideographs makes token counts and
    # SequenceMatcher ratios behave like character-level matching instead of comparing
    # one unsegmented run of characters against another.
    return re.sub(r'\s+', ' ', space_cjk(s)).strip()


class EntityMatcher:
    def __init__(self, mode='string', threshold=0.6, use_type_nodes=True):
        assert mode in ('string', 'embedding'), mode
        self.mode = mode
        self.threshold = threshold
        self.use_type_nodes = use_type_nodes
        self.warned_no_entity_links = False

    # ------------------------------------------------------------------
    def match(self, text_graph, visual_graph, text_feats=None, visual_feats=None):
        """Returns A_tv [Nt, Nv] float32 (1.0 = matched) and the matched pairs."""
        nt, nv = len(text_graph.nodes), len(visual_graph.nodes)
        A = np.zeros((nt, nv), dtype=np.float32)
        pairs = []

        t_norm = [normalize(x) for x in text_graph.nodes]
        v_norm = [normalize(x) for x in visual_graph.nodes]
        t_gen = text_graph.generic_slice
        v_gen = visual_graph.generic_slice
        # The global nodes v_g are summaries, not entities: the visual one embeds the
        # whole caption, so leaving it in makes every textual entity substring-match it
        # and floods A_tv with links that carry no alignment signal.
        skip_t = {text_graph.global_index} | (set(range(*t_gen)) if t_gen[0] >= 0 else set())
        skip_v = {visual_graph.global_index} | (set(range(*v_gen)) if v_gen[0] >= 0 else set())

        for i in range(nt):
            if i in skip_t:
                continue                      # generic nodes are linked separately
            for j in range(nv):
                if j in skip_v:
                    continue
                score = self._score(t_norm[i], v_norm[j],
                                    None if text_feats is None else text_feats[i],
                                    None if visual_feats is None else visual_feats[j])
                if score >= self.threshold:
                    A[i, j] = score
                    pairs.append((i, j, score))

        if self.use_type_nodes:
            # Sec.3.2.2: the four generic type nodes bridge the two graphs structurally
            for gi, name in enumerate(GENERIC_TYPES):
                ti, vj = t_gen[0] + gi, v_gen[0] + gi
                if ti < nt and vj < nv:
                    A[ti, vj] = 1.0
                    pairs.append((ti, vj, 1.0))

        # Zero *entity* links (as opposed to the four structural type bridges) is almost
        # always a configuration problem rather than a fact about the post, so say so once
        # instead of letting a bridge-only cache look like a valid alignment result.
        t_ent = [i for i in range(nt) if i not in skip_t]
        v_ent = [j for j in range(nv) if j not in skip_v]
        real_pairs = [p for p in pairs if p[2] < 0.999]        # ignore the type bridges
        if self.mode == 'string' and t_ent and v_ent and not real_pairs:
            if not self.warned_no_entity_links:
                texts = [text_graph.nodes[i] for i in t_ent[:3]]
                objs = [visual_graph.nodes[j] for j in v_ent[:3]]
                cjk_t = any(is_cjk(x) for x in texts)
                latin_v = all(not is_cjk(x) for x in objs)
                reason = ('textual entities are Chinese while visual entity names are not '
                          '(detector labels were not localised; use lang="zh" or a '
                          'translated label vocabulary)') if cjk_t and latin_v else \
                    ('no string overlap at all; check that the visual side carries real '
                     'object names (an "unknown"/grid proposal vocabulary matches nothing)')
                print(f'[warn] {len(pairs)} cross-modal links are all generic type bridges: '
                      f'{reason}. textual={texts} visual={objs}')
                self.warned_no_entity_links = True
        return A, pairs

    def _score(self, a: str, b: str, fa, fb):
        if not a or not b:
            return 0.0
        if self.mode == 'embedding' and fa is not None and fb is not None:
            return float(np.dot(fa, fb) / (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-8))
        if a == b:
            return 1.0
        if a in b or b in a:
            # containment is only strong evidence when the shorter side is a sequence of
            # whole tokens of the longer one; otherwise "man" would match "omnipresent"
            shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
            toks = shorter.split()
            if len(toks) >= 2 or (toks and toks[0] in longer.split()):
                return 0.85
        return difflib.SequenceMatcher(None, a, b).ratio()
