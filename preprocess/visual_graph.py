"""Sec.3.1.1 - Multimodal Entity Graph Construction (visual side).

G_v = (V_I, E_I): object nodes with their region features v_obj, OCR nodes v_ocr from the
image text, a global node v_g for the whole image, and edges E_I carrying the relation
predicate between two regions.

DEVIATION: the paper uses a trained scene-graph generator (Mask-RCNN + relation head) on
data we do not have. Here DETR provides the boxes/labels and the predicates are derived
from box geometry (containment / overlap / relative position). The node and edge
*vocabulary and tensor layout are unchanged*, so swapping in a real SGG model only
requires replacing `_propose` + `_box_relations`.
"""
import numpy as np
import torch

from .text_graph import EntityGraph, GENERIC_TYPE_ID, GENERIC_TYPES

VISUAL_RELATIONS = [
    '<pad>', '<self>', 'unknown', 'holding', 'wearing', 'riding', 'carrying',
    'eating', 'drinking', 'looking_at', 'touching', 'biting', 'sitting_on',
    'standing_on', 'lying_on', 'walking_on', 'running_on', 'jumping_from',
    'flying_from', 'turning', 'leaning_on', 'talking_to', 'contained_in',
    'covering', 'washing', 'cutting', 'closest_to', 'above', 'below', 'left_of',
    'right_of', 'overlapping', 'global', 'type',
]
VISUAL_REL2ID = {r: i for i, r in enumerate(VISUAL_RELATIONS)}

# DETR publishes COCO class names in English. For a Chinese post (Weibo16) the entity
# surface strings must therefore be translated, otherwise textual and visual entities can
# never be linked by string matching. Anything not covered here is kept in English and
# reported by EntityMatcher's cross-lingual warning rather than silently dropped.
COCO_ZH = {
    'person': '\u4eba', 'bicycle': '\u81ea\u884c\u8f66', 'car': '\u6c7d\u8f66',
    'motorcycle': '\u6469\u6258\u8f66', 'airplane': '\u98de\u673a', 'bus': '\u516c\u5171\u6c7d\u8f66',
    'train': '\u706b\u8f66', 'truck': '\u5361\u8f66', 'boat': '\u8239',
    'traffic light': '\u7ea2\u7eff\u706f', 'fire hydrant': '\u6d88\u9632\u6813',
    'stop sign': '\u505c\u8f66\u6807\u5fd7', 'bench': '\u957f\u51f3', 'tv': '\u7535\u89c6\u673a',
    'laptop': '\u7b14\u8bb0\u672c\u7535\u8111', 'chair': '\u6905\u5b50',
    'dining table': '\u9910\u684c', 'potted plant': '\u76c6\u683d', 'bed': '\u5e8a',
    'clock': '\u65f6\u949f', 'couch': '\u6c99\u53d1', 'keyboard': '\u952e\u76d8',
    'cell phone': '\u624b\u673a', 'book': '\u4e66', 'scissors': '\u526a\u5200',
}

# NOTE: the four generic type nodes deliberately keep their English surface forms in
# both graphs (see GENERIC_TYPES); they are bridged by index, never by string, so
# localising them would only add vocabulary noise.


def _iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    return inter / max(1e-6, a1 + a2 - inter)


def _box_relations(b1, b2):
    """Geometric stand-in for the SGG relation head. b* = [x1,y1,x2,y2] in pixels."""
    c1x, c1y = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2
    c2x, c2y = (b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    if a2 > 0 and a1 / a2 > 1.8 and b1[0] >= b2[0] - 2 and b1[2] <= b2[2] + 2 \
            and b1[1] >= b2[1] - 2 and b1[3] <= b2[3] + 2:
        return 'contained_in'
    if _iou(b1, b2) > 0.2:
        return 'overlapping'
    dx, dy = c1x - c2x, c1y - c2y
    if abs(dy) > abs(dx):
        return 'above' if dy < 0 else 'below'
    return 'left_of' if dx < 0 else 'right_of'


class VisualGraphBuilder:
    def __init__(self, detector='facebook/detr-resnet-50', ocr_backend='none',
                 top_k=10, max_nodes=16, use_region_feats=True, device='cpu',
                 box_threshold=0.5, proposal_fallback=True, lang='en'):
        self.lang = lang
        self.top_k = top_k
        self.max_nodes = max_nodes
        self.use_region_feats = use_region_feats
        self.ocr_backend = ocr_backend
        self.device = torch.device(device if (device == 'cpu' or torch.cuda.is_available())
                                   else 'cpu')
        self._tf = None
        self.detector = None
        self.region_model = None
        self.region_transform = None
        self.ocr = None
        # smoke-test affordance: emit spatial proposals when the detector finds nothing,
        # so the object-node / region-feature path is still exercised. Never triggers on
        # real photos, where DETR does detect objects.
        self.proposal_fallback = proposal_fallback
        self._warned_fallback = False
        if detector not in (None, 'none'):
            self._load_detector(detector, box_threshold)
        if ocr_backend == 'easyocr':
            self._load_ocr()

    # ------------------------------------------------------------------
    def _load_detector(self, name, box_threshold):
        from transformers import DetrImageProcessor, DetrForObjectDetection
        self.box_threshold = box_threshold
        self._tf = DetrImageProcessor.from_pretrained(name)
        self.detector = DetrForObjectDetection.from_pretrained(name).to(self.device).eval()

    def _load_region_backbone(self):
        """ResNet50 trunk reused for ROI crops (v_obj) and the global feature (v_g)."""
        from torchvision.models import ResNet50_Weights, resnet50
        from torchvision.transforms import CenterCrop, Compose, Normalize, ToTensor
        weights = ResNet50_Weights.IMAGENET1K_V1
        m = resnet50(weights=weights).to(self.device).eval()
        self.region_model = torch.nn.Sequential(*list(m.children())[:-2])  # -> [B,2048,h,w]
        # weights.meta['normalization'] is not stable across torchvision versions, so
        # pull mean/std out of the official transform instead.
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        try:
            for tr in getattr(weights.transforms(), 'transforms', []):
                if isinstance(tr, Normalize):
                    mean, std = list(tr.mean), list(tr.std)
        except Exception:  # noqa: BLE001
            pass
        self.region_transform = Compose([CenterCrop(224), ToTensor(),
                                         Normalize(mean, std)])
        self.resize = CenterCrop(224)

    def _load_ocr(self):
        import easyocr
        self.ocr = easyocr.Reader(['en'], gpu=(self.device.type == 'cuda'), verbose=False)

    def _propose(self, image):
        """DETR object proposals: list of (label, score, box_pixels)."""
        inputs = self._tf(images=image, return_tensors='pt').to(self.device)
        with torch.no_grad():
            out = self.detector(**inputs)
        # boxes must be rescaled to the original pixel grid. transformers>=4.55 no
        # longer puts `_sizes` in the processor output, so take the size from the PIL
        # image itself; target_sizes is (height, width) like transformers expects.
        W, H = image.size
        results = self._tf.post_process_object_detection(
            out, threshold=self.box_threshold, target_sizes=[(H, W)])[0]
        items = []
        for score, label_id, box in zip(results['scores'].tolist(),
                                        results['labels'].tolist(),
                                        results['boxes'].tolist()):
            label = self.detector.config.id2label[label_id]
            items.append((label, score, box))
        items.sort(key=lambda x: -x[1])
        if items or not self.proposal_fallback:
            return items[:self.top_k]
        if not self._warned_fallback:
            print('[warn] DETR returned no proposals; falling back to a 2x2 spatial '
                  'grid. Expected on synthetic renders, and harmless, but on real '
                  'datasets it means the detector is misconfigured.')
            self._warned_fallback = True
        return self._grid_proposals(image)

    def _grid_proposals(self, image):
        """Class-agnostic 2x2 spatial proposals, used only when DETR finds nothing."""
        W, H = image.size
        out = []
        for r in range(2):
            for c in range(2):
                box = [c * W / 2, r * H / 2, min(W, (c + 1) * W / 2), min(H, (r + 1) * H / 2)]
                out.append(('unknown', 0.0, box))
        return out[:self.top_k]

    def _region_feats(self, image, boxes):
        # an empty proposal list would otherwise build a shape-(0,) tensor and break
        # the elementwise scaling below
        if len(boxes) == 0:
            return np.zeros((0, 2048), dtype=np.float32)
        if not self.use_region_feats:
            return np.zeros((len(boxes), 2048), dtype=np.float32)
        if self.region_model is None:
            self._load_region_backbone()
        from torchvision.ops import roi_align
        img = self.region_transform(image.convert('RGB')).to(self.device)  # [3,224,224]
        W, H = image.size
        b = torch.tensor(boxes, dtype=torch.float32, device=self.device).reshape(-1, 4)
        b = b * torch.tensor([224 / W, 224 / H, 224 / W, 224 / H], device=self.device)
        b[:, 2] = b[:, 2].clamp(max=224)
        b[:, 3] = b[:, 3].clamp(max=224)
        feats = self.region_model(img.unsqueeze(0))                  # [1,2048,7,7]
        pooled = roi_align(feats, [b], output_size=(1, 1), spatial_scale=1.0)
        return pooled[:, :, 0, 0].squeeze(-1).squeeze(-1).detach().cpu().numpy().astype(np.float32)

    def _global_feat(self, image):
        if self.region_model is None:
            self._load_region_backbone()
        img = self.region_transform(image.convert('RGB')).to(self.device)
        with torch.no_grad():
            fmap = self.region_model(img.unsqueeze(0))
        return fmap.mean(dim=(2, 3))[0].cpu().numpy().astype(np.float32)

    def _ocr_nodes(self, image_path):
        if self.ocr is None:
            return []
        out = []
        for res in self.ocr.readtext(str(image_path)):
            (x1, y1), _, (x2, y2), _ = res[0]
            txt, conf = res[1], res[2]
            if conf < 0.3 or not txt.strip():
                continue
            out.append((txt.strip()[:64], (x1, y1, x2, y2)))
        return out[:4]

    # ------------------------------------------------------------------
    def build(self, image, image_path=None, caption=''):
        """`image` is a PIL.Image; `image_path` is only needed for the OCR tool."""
        from PIL import Image
        if isinstance(image, (str, bytes)):
            image = Image.open(image).convert('RGB')
        g = EntityGraph()
        boxes = []

        proposals = self._propose(image) if self.detector is not None else []
        for label, score, box in proposals:
            if len(g.nodes) >= self.max_nodes - 6:
                break
            # the *type* is inferred from the English COCO label, only the node surface
            # string is localised, so switching language cannot break _type_of()
            idx = g.add(self._surface(label), self._type_of(label))
            boxes.append(box)
            g.edges.append((idx, idx, VISUAL_REL2ID['<self>']))

        # OCR nodes v_ocr, attached to the object that contains them
        ocr_items = self._ocr_nodes(image_path) if image_path else []
        for txt, ob in ocr_items:
            idx = g.add(f'text in image: {txt}', 'misc')
            boxes.append(list(ob))
            host = next((i for i, b in enumerate(boxes[:-1])
                         if _iou(b, list(ob)) > 0.5), None)
            g.add_edge(idx, host if host is not None else len(boxes) - 1, 'contained_in')

        # global node v_g
        g.global_index = g.add(self._global_text(caption, proposals), 'misc')
        for i in range(len(g.nodes) - 1):
            g.add_edge(i, g.global_index, 'global')

        # pairwise geometric / SGG predicates
        for i, bi in enumerate(boxes):
            for j, bj in enumerate(boxes):
                if i >= j or i >= len(g.nodes) or j >= len(g.nodes):
                    continue
                g.edges.append((i, j, VISUAL_REL2ID[_box_relations(bi, bj)]))

        self.append_generic_nodes(g)

        n = min(len(g.nodes), self.max_nodes)
        vis = np.zeros((self.max_nodes, 2048), dtype=np.float32)
        gf = self._global_feat(image)
        if n:
            crop_feats = self._region_feats(image, boxes[:n])
            k = min(len(crop_feats), n)
            vis[:k] = crop_feats[:k]
            vis[n - 1] = gf          # the global node carries the whole-image feature
        A, R = g.adjacency(self.max_nodes)
        return g, A, R, vis, boxes[:n], gf

    def _surface(self, label: str) -> str:
        """Node text in the post's language; English COCO names are kept when unknown."""
        if self.lang != 'zh':
            return label
        return COCO_ZH.get(label.lower(), label)

    def _global_text(self, caption, proposals):
        """v_g summary text, in the post's language (it is BERT-encoded as a node)."""
        labels = ', '.join(sorted({self._surface(p[0]) for p in proposals})) \
            or ('未识别的场景' if self.lang == 'zh' else 'unrecognised scene')
        head = (caption or '').strip()[:80]
        if self.lang == 'zh':
            base = f'\u56fe\u7247\u4e2d\u5305\u542b {labels}'
            return base + (f'\uff1b\u914d\u6587: {head}' if head else '')
        return f'the image shows {labels}' + (f'; caption: {head}' if head else '')

    @staticmethod
    def _type_of(label):
        """Rough COCO-label -> generic type mapping for the cross-modal type nodes."""
        L = (label or '').lower()
        if any(k in L for k in ('person', 'bicycle rider')):
            return 'person'
        if any(k in L for k in ('building', 'mountain', 'sea', 'sky', 'road', 'bridge',
                                'floor', 'field', 'tree')):
            return 'location'
        if any(k in L for k in ('truck', 'bus', 'train', 'airplane', 'boat', 'bench')):
            return 'organization'
        return 'misc'

    def append_generic_nodes(self, g: EntityGraph):
        start = len(g.nodes)
        members = [[] for _ in GENERIC_TYPES]
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
