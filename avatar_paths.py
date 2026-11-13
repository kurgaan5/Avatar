import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = Path(os.environ.get('AVATAR_RUNTIME', ROOT / '.runtime')).resolve()
MODELS = Path(os.environ.get('AVATAR_MODELS', RUNTIME / 'models')).resolve()
DH = Path(os.environ.get('AVATAR_DH', RUNTIME / 'DH_live')).resolve()
LIVEPORTRAIT = Path(os.environ.get('AVATAR_LIVEPORTRAIT', RUNTIME / 'LivePortrait')).resolve()
