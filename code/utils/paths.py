from pathlib import Path

# --- locate project root automatically ---
# assume this file always lives in meningioma_radiomics/code/utils/
ROOT = Path(__file__).resolve().parents[2]

# --- key folders ---
DATA_DIR = ROOT / "data"

# subfolders under data
PREPROC_DIR = DATA_DIR / "0_preprocessing"
FEATEX_DIR = DATA_DIR / "1_feature_extraction"
MODELING_DIR = DATA_DIR / "2_modeling"

# subfolders under feature extraction dir
PYRAD_DIR = FEATEX_DIR / "1_pyradiomics"
COLLAGE_DIR = FEATEX_DIR / "2_collage"

# --- USER DEFINED DIRS & FILES OF INTEREST ---
MRIS_DIR = PREPROC_DIR / "7b_COMPLETED_PREPROCESSED"
SEGS_DIR = PREPROC_DIR / "all_smooth_segs_02-08-25"
LABELS_FILE = PREPROC_DIR / "labels" / "MeningiomaBiomarkerData.csv"

# --- ensure directories exist ---
for p in [
    PREPROC_DIR,
    FEATEX_DIR,
    MODELING_DIR,
    MRIS_DIR,
    SEGS_DIR,
    PYRAD_DIR,
    COLLAGE_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)
