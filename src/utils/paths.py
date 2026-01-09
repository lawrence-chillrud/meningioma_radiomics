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

# --- USER DEFINED DIRS, FILES, GLOBAL VARS OF INTEREST ---
MRIS_DIR = PREPROC_DIR / "7_COMPLETED_PREPROCESSED"
SEGS_DIR = PREPROC_DIR / "8_ALL_SMOOTH_SEGS_02-08-25"
LABELS_FILE = PREPROC_DIR / "0_LABELS" / "MeningiomaBiomarkerData.csv"
METADATA_FILE = (
    PREPROC_DIR / "0_LABELS" / "radiomics_cohort_06-09-2025_w_demographics_clean.csv"
)
PULSES = ["T1_POST", "FLAIR", "DIFFUSION", "ADC"]
ROIS = [1, 3, 4, 5, 6, 13, 15, 16, 156, 22]
PYRAD_FILE = PYRAD_DIR / "b_aggregated_pyradiomics" / "12-08-2025_19-00-57_features.csv"

# --- ensure directories exist ---
for p in [
    PREPROC_DIR,
    FEATEX_DIR,
    MODELING_DIR,
    PYRAD_DIR,
    COLLAGE_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)
