"""Classical CPU denoisers inspired by quantum mechanics; no quantum SDK needed."""

from .dequip import DeQuIPConfig, dequip_denoise
from .qab import QABImageConfig, QABSignalConfig, qab_image, qab_signal

__all__ = ["DeQuIPConfig", "dequip_denoise", "QABImageConfig", "QABSignalConfig", "qab_image", "qab_signal"]
__version__ = "0.1.0"

