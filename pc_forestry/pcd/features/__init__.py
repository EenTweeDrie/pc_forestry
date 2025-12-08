from .base import VoxelFeature
from .registry import registry, register_feature
from . import builtin  # noqa: F401 - важен побочный эффект регистрации
