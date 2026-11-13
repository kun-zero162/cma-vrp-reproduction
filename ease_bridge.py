"""Reuse the EASE Expert scaffolding (utils / metrics) without duplicating it.

Loaded by absolute path so this package stays importable from anywhere and never
shadows the local `utils`/`engine` module names.
"""
import importlib.util
import os

_EXPERT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Expert')
)


def _load(mod_name: str, file_name: str):
    path = os.path.join(_EXPERT_DIR, file_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Cannot locate EASE scaffolding file: {path}')
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_expert_module(file_name: str, mod_name: str | None = None):
    """Load an EASE Expert module by path.

    Never touches `sys.path`: appending Expert/ there would let `configs`, `models` and
    `datasets` resolve to EASE's packages instead of CMAVRP's.
    """
    return _load(mod_name or f"ease_{os.path.basename(file_name).replace('.', '_')}", file_name)


ease_utils = _load('ease_utils', 'utils.py')
ease_engine = _load('ease_engine', 'engine.py')

set_seed = ease_utils.set_seed
get_logger = ease_utils.get_logger
log_config_info = ease_utils.log_config_info
get_optimizer = ease_utils.get_optimizer
get_scheduler = ease_utils.get_scheduler
cal_params_flops = ease_utils.cal_params_flops
save_results = ease_utils.save_results
load_json_data = ease_utils.load_json_data
Recorder = ease_utils.Recorder
Averager = ease_utils.Averager
data2gpu = ease_utils.data2gpu
calculate_metrics = ease_engine.calculate_metrics
