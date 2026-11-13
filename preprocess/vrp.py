"""Sec.3.1.2 - Truth-Driven CoT-based Visual Reasoning Prompting (VRP).

Two prompting layers, run on a text LLM for textual entities and an LVLM for visual
entities, *without any fine-tuning*:

layer 1 (rationale): "For each object, consider the rationale for how its visual features
                     justify whether the claim contains misinformation."
layer 2 (summary)  : "Summarize the corresponding visual features of each object for each
                     rationale."

The output reasoning texts are BERT-encoded into VRF_t / VRF_v by build_dataset.py.
"""
import os
import re
import sys

_CMAVRP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SYSTEM = ('You are a multimodal misinformation analyst. You reason strictly about the '
          'visual evidence of entities mentioned in a social-media post.')

# ---- layer 1: dual-layer CoT rationale (paper Fig.2b) ----
PROMPT_RATIONALE = '''{system}

Claim: "{claim}"
Veracity (ground truth, used only to steer the reasoning): {label}
Textual entities extracted from the post: {text_entities}
Visual entities extracted from the image: {visual_entities}

For each object, consider the rationale for how its visual features justify whether the \
claim contains misinformation.

Output:
'''

PROMPT_RATIONALE_NOGT = '''{system}

Claim: "{claim}"
Textual entities extracted from the post: {text_entities}
Visual entities extracted from the image: {visual_entities}

For each object, consider the rationale for how its visual features justify whether the \
claim contains misinformation.

Output:
'''

# ---- layer 2: per-object visual summary ----
PROMPT_SUMMARIZE = '''{system}

Below are rationales about the visual features of each entity for the claim: "{claim}"
(ground-truth veracity: {label}).

{rationales}

Summarize the corresponding visual features of each object for each rationale.

Output:
'''

_TEXT_HEAD = re.compile(r'\(\s*Textual Entit\w*\s*\)', re.I)
_VISUAL_HEAD = re.compile(r'\(\s*Visual Entit\w*\s*\)', re.I)
_ITEM = re.compile(r'^\s*([^:]{1,60})\s*[:：]\s*(.+)$')


def format_entities(names):
    return '[' + ', '.join(f'{n}' for n in names) + ']' if names else '[]'


def parse_vrf(output, text_names, visual_names):
    """Map '(Textual Entity) X: ... (Visual Entity) Y: ...' back onto node indices.

    Returns (vrf_t, vrf_v) lists aligned with the node lists; unmatched nodes fall back to
    the sentence that mentions them so that no node is left without a reasoning feature.
    """
    out_t = [''] * len(text_names)
    out_v = [''] * len(visual_names)
    if not output:
        return out_t, out_v

    # split into a textual block and a visual block
    heads = [(m.start(), 'v' if _VISUAL_HEAD.match(output, m.start()) else 't')
             for m in re.finditer(r'\(\s*(Textual|Visual) Entit\w*\s*\)', output, re.I)]
    blocks = []
    for k, (pos, kind) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(output)
        blocks.append((kind, output[pos:end]))

    t_norm = {n.lower().strip(): i for i, n in enumerate(text_names)}
    v_norm = {n.lower().strip(): i for i, n in enumerate(visual_names)}
    for kind, chunk in blocks:
        for line in chunk.splitlines()[1:]:
            m = _ITEM.match(line.strip().lstrip('-*•{ '))
            if not m:
                continue
            name, body = m.group(1).strip(), m.group(2).strip().rstrip('},] ')
            key = name.lower()
            table = t_norm if kind == 't' else v_norm
            if key in table:
                idx = table[key]
            else:  # tolerate "Man_0", "Protester." style references
                base = re.sub(r'_\d+$|[.\s]+$', '', key)
                idx = next((v for k, v in table.items() if k.startswith(base) or base.startswith(k)),
                           None)
                if idx is None:
                    continue
            slot = out_t if kind == 't' else out_v
            slot[idx] = (slot[idx] + ' ' + body).strip()[:600]

    for i, (slot, names) in enumerate(((out_t, text_names), (out_v, visual_names))):
        for j in range(len(slot)):
            if not slot[j]:
                slot[j] = f'{names[j]}: no explicit visual evidence available for this entity.'
    return out_t, out_v


class VisualReasoningPrompter:
    """backend='mock' runs offline; 'hf' loads Mistral-7B + LLaVA; 'agent' calls the EASE Agent LLM."""

    def __init__(self, backend='mock', llm_text='', lvlm_visual='', device='cpu',
                 max_new_tokens=256):
        self.backend = backend
        self.llm_text = llm_text
        self.lvlm_visual = lvlm_visual
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._pipe = None
        self._lvlm = None
        self._lvlm_proc = None
        self._agent_model = None

    # ------------------------------------------------------------------
    def _load_hf(self):
        if self._pipe is not None or self.backend != 'hf':
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        print(f'[vrp] loading {self.llm_text} ...')
        self._pipe = pipeline('text-generation',
                              model=self.llm_text,
                              tokenizer=AutoTokenizer.from_pretrained(self.llm_text),
                              torch_dtype=torch.float16 if self.device != 'cpu' else torch.float32,
                              device_map='auto' if self.device != 'cpu' else None)

    @staticmethod
    def _mock(prompt):
        """Deterministic stand-in so the whole chain runs without a 13B model on disk."""
        return '(Textual Entity) unspecified: no LLM available.\n'

    def _load_lvlm(self):
        if self._lvlm is not None or self.backend != 'hf':
            return
        import torch
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        print(f'[vrp] loading {self.lvlm_visual} ...')
        self._lvlm_proc = LlavaNextProcessor.from_pretrained(self.lvlm_visual)
        self._lvlm = LlavaNextForConditionalGeneration.from_pretrained(
            self.lvlm_visual,
            torch_dtype=torch.float16 if self.device != 'cpu' else torch.float32,
            device_map='auto' if self.device != 'cpu' else None)

    def _load_agent(self):
        """Reuses the EASE Agent's LLM abstraction (config/api_keys.yaml)."""
        if self._agent_model is not None or self.backend != 'agent':
            return
        agent_dir = os.path.join(os.path.dirname(_CMAVRP_DIR), 'Agent')
        if agent_dir not in sys.path:
            sys.path.insert(0, agent_dir)
        from agent.common.modeling import make_model  # noqa: E402
        self._agent_model = make_model('gpt_4o')

    # ------------------------------------------------------------------
    def _llm(self, prompt):
        if self.backend == 'mock':
            return self._mock(prompt)
        if self.backend == 'agent':
            self._load_agent()
            return self._agent_model.generate([{'role': 'user', 'content': prompt}]).content
        self._load_hf()
        return self._pipe(prompt, max_new_tokens=self.max_new_tokens,
                          do_sample=False, return_full_text=False)[0]['generated_text']

    def _lvlm(self, prompt, image_path):
        if self.backend in ('mock', 'agent'):
            # 'agent' has no LVLM wired here; the EASE Agent tools cover that separately
            return self._mock(prompt + ' [visual]')
        self._load_lvlm()
        from PIL import Image
        conv = [{'role': 'user', 'content': [
            {'type': 'image', 'image': Image.open(image_path).convert('RGB')},
            {'type': 'text', 'text': prompt}]}]
        inputs = self._lvlm_proc.from_conversations([conv], tokenize=True,
                                                    return_tensors='pt').to(self._lvlm.device)
        gen = self._lvlm.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self._lvlm_proc.decode(gen[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    def reason(self, claim, text_names, visual_names, label=None, image_path=None):
        """Returns (vrf_texts_t, vrf_texts_v)."""
        gt = {0: 'real', 1: 'fake'}.get(int(label), 'unknown') if label is not None else 'unknown'
        tpl = PROMPT_RATIONALE if label is not None else PROMPT_RATIONALE_NOGT
        kw = dict(system=SYSTEM, claim=claim, label=gt,
                  text_entities=format_entities(text_names),
                  visual_entities=format_entities(visual_names))
        p1 = tpl.format(**kw)          # extra keys are ignored by the no-ground-truth variant

        if self.backend == 'mock':
            out1 = self._mock_rationale(text_names, visual_names, claim, gt)
        else:
            out1 = self._llm(p1)
        r_t, r_v = parse_vrf(out1, text_names, visual_names)

        joined = '\n'.join(['(Textual Entity) ' + f'{n}: {t}' for n, t in zip(text_names, r_t)]
                           + ['(Visual Entity) ' + f'{n}: {t}' for n, t in zip(visual_names, r_v)])
        p2 = PROMPT_SUMMARIZE.format(system=SYSTEM, claim=claim, label=gt, rationales=joined)
        if self.backend == 'mock':
            out2 = self._mock_summary(r_t, r_v, text_names, visual_names)
        elif image_path and self.backend == 'hf':
            out2 = self._lvlm(p2, image_path)
        else:
            out2 = self._llm(p2)
        s_t, s_v = parse_vrf(out2, text_names, visual_names)
        return s_t, s_v

    # ---------------- offline mock reasoning ----------------
    @staticmethod
    def _mock_rationale(text_names, visual_names, claim, gt):
        lines = []
        for n in text_names:
            if n.lower() in ('misc', 'person', 'location', 'organization'):
                continue
            lines.append(f'(Textual Entity) {n}: we must check whether the image actually '
                         f'shows "{n}" in the state the claim describes.')
        for n in visual_names:
            if n.lower() in ('misc', 'person', 'location', 'organization'):
                continue
            lines.append(f'(Visual Entity) {n}: the visible appearance and position of the '
                         f'{n} either corroborates or contradicts the claim.')
        return '\n'.join(lines)

    @staticmethod
    def _mock_summary(r_t, r_v, text_names, visual_names):
        lines = []
        for n, t in zip(text_names, r_t):
            lines.append(f'(Textual Entity) {n}: {{ {t[:110]} }}')
        for n, t in zip(visual_names, r_v):
            lines.append(f'(Visual Entity) {n}: {{ {t[:110]} }}')
        return '\n'.join(lines)
