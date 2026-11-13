import os

_CMAVRP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # CMAVRP/
_ROOT_DIR = os.path.dirname(_CMAVRP_DIR)                                     # EASE/


def _resolve_bert(hf_id='google-bert/bert-base-uncased'):
    """Use a locally downloaded copy when present, otherwise the HF hub id."""
    name = hf_id.split('/')[-1]
    local = os.path.join(_CMAVRP_DIR, 'bert', name)
    return local if os.path.isdir(local) else hf_id


class setting_config:
    """CMA-VRP (MM'25) reproduction settings.

    Paper: "Entity Graph Alignment and Visual Reasoning for Multimodal Fake News Detection"
    Guoyi Li, Die Hu, Xiaomeng Fu, Qirui Tang, Yulei Wu, Xiaodan Zhang, Honglei Lyu.
    ACM MM 2025, DOI 10.1145/3746027.3754591.
    Values marked [S4] come from the Sec.4 implementation details.
    """

    network = 'cmavrp'

    # ---------------- data ----------------
    datasets = 'twitter'                       # twitter | pheme | weibo
    raw_dir = os.path.join(_CMAVRP_DIR, 'data')
    data_path = ''                             # filled by train.py
    cache_path = ''                            # filled by train.py
    max_text_nodes = 32
    max_visual_nodes = 16
    top_k_objects = 10                         # [S3.1.1] "top k objects ranked by scores"
    ocr_max_nodes = 4
    use_region_feats = True                    # X_v also gets ResNet region features

    # ---------------- encoders [S4] ----------------
    # [S4] specifies BERT-base-uncased + en_core_web_trf, i.e. an English pipeline, but
    # Weibo16 is Chinese -- so the Chinese configuration is spelled out here rather than
    # silently reusing the English one (see preprocess/lang.py).
    lang = 'auto'                             # auto | en | zh
    bert_path = _resolve_bert()               # English node encoder
    bert_path_zh = _resolve_bert('google-bert/bert-base-chinese')
    spacy_model = 'en_core_web_trf'
    spacy_model_zh = 'zh_core_web_sm'         # CPU-friendly; _trf is the GPU variant
    llm_text = 'mistralai/Mistral-7B-Instruct-v0.2'
    lvlm_visual = 'llava-hf/llava-v1.6-vicuna-13b-hf'
    detector = 'facebook/detr-resnet-50'
    ocr_backend = 'none'                       # easyocr | none  (Rosetta is not public)
    vrp_backend = 'mock'                       # mock | hf | agent
    device = 'auto'                            # auto | cuda | cpu

    # ---------------- graph fusion, Sec.3.2.2 ----------------
    model_config = {
        'node_dim': 768,                        # BERT hidden size
        'region_dim': 2048,                     # ResNet50 crop feature
        'gnn_dim': 300,
        'n_fusion_layers': 2,                   # [S4] L = 2
        'ffn_dim': 600,
        'dropout': 0.2,
        'readout': 'mean',                      # mean | max | mean_max
        'max_text_relations': 40,
        'max_visual_relations': 40,
    }

    # ---------------- contrastive learning, Sec.3.2.1 ----------------
    cl_config = {
        'tau': 0.07,                            # InfoNCE temperature
        'p_m': 0.2,                             # [S4] entity masking probability
        'theta_init': 0.5,                      # [S3.2.1] learnable smoothing, init 0.5
        'symmetric_inter': False,               # Eq.(3) is written in one direction only
        'detach_soft_target': True,             # S acts as a stop-gradient teacher target
    }

    # ---------------- VSCA + ACCIL, Sec.3.3 ----------------
    adv_config = {
        'latent_dim': 300,
        'hidden_dim': 512,
        'gp_lambda': 10.0,
        'sinkhorn_eps': 0.1,
        'sinkhorn_iters': 50,
        'cyc_lambda': 10.0,
        'ot_lambda': 1.0,
        'use_grl': True,                        # reuse EASE models/layers.ReverseLayerF
        'grl_alpha_max': 1.0,
    }

    # ---------------- loss weights [S4] ----------------
    lambda_cl = 0.5                            # lambda
    beta_adv = 0.1                             # beta

    # ---------------- ablation switches (Table 2) ----------------
    ablation = {
        'use_vrf': True,
        'use_gcl_inter': True,
        'use_gcl_intra': True,
        'use_cmgaf': True,
        'use_accil': True,
        'use_vsca': True,
    }

    # ---------------- training [S4] ----------------
    criterion = 'CrossEntropy'
    num_classes = 2
    seed = 3759
    batch_size = 32
    epochs = 50                                # early stopping after 5/50 epochs
    early_stop = 5
    gpu_id = '0'
    use_cuda = True
    num_workers = 0                            # Windows + tiny datasets: keep 0
    threshold = 0.5
    print_interval = 5
    amp = False
    eval_mode = False

    opt = 'Adam'
    lr = 0.001                                 # [S4]
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 1e-4
    amsgrad = False

    sch = 'StepLR'
    step_size = 5
    gamma = 0.5                                # [S4] decay 0.5
    last_epoch = -1
