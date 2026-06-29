from .distillation_cfg import *
from .exporter import export_policy_as_jit, export_policy_as_onnx
from .rl_cfg import *
from .rnd_cfg import RslRlRndCfg
from .symmetry_cfg import RslRlSymmetryCfg


def __getattr__(name: str):
    if name == "RslRlVecEnvWrapper":
        from .vecenv_wrapper import RslRlVecEnvWrapper

        return RslRlVecEnvWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
