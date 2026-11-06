"""Project-relative defaults; original data/checkpoints stay outside Git."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIVA_REPO = Path(os.environ.get('DIVA_REPO', ROOT / 'external/DIVA')).resolve()
DIVA_TEST = DIVA_REPO / 'DIVA test code for image denoising'
TRAINING_NOTEBOOK = ROOT / 'methods/diva/upstream/DIVA_training.ipynb'
TRAIN400_DIR = Path(os.environ.get('TRAIN400_DIR', ROOT / 'data/Train400_subset64')).resolve()
