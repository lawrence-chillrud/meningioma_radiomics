from pathlib import Path

# --- locate project root automatically ---
# assume this file always lives in meningioma_radiomics/code/utils/
ROOT = Path(__file__).resolve().parents[2]

# --- key folders ---
DATA_DIR = ROOT / "data"

# subfolders under data
DATA_PREPROC = DATA_DIR / "0_preprocessing"
DATA_FEATEX = DATA_DIR / "1_feature_extraction"
DATA_MODEL = DATA_DIR / "2_modeling"

# optional: ensure directories exist
for p in [DATA_PREPROC, DATA_FEATEX, DATA_MODEL]:
    p.mkdir(parents=True, exist_ok=True)
