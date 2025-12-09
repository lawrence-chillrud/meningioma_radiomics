from datetime import datetime
import joblib
import logging
import pandas as pd
from src.utils import *
from tqdm import tqdm

# Set up data directories
A_DIR = PYRAD_DIR / "a_raw_pyradiomics"
RAW_PYRAD_DIR = sorted([d for d in A_DIR.iterdir() if d.is_dir()])[-1]
OUTPUT_DIR = PYRAD_DIR / "b_aggregated_pyradiomics"
OUTPUT_FP = OUTPUT_DIR / f"{RAW_PYRAD_DIR.name}_features.csv"
FEAT_FILES = sorted([f for f in RAW_PYRAD_DIR.iterdir() if f.name.endswith(".joblib")])

# Setup logfile
TIMESTAMP = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
LOGFILE = OUTPUT_DIR / "logging.txt"
logging.basicConfig(
    filename=LOGFILE,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info(f"<>" * 40)
logging.info(f"Log entry for {Path(__file__).name} ran at {TIMESTAMP}")
logging.info(f"\tRAW_PYRAD_DIR in use: {RAW_PYRAD_DIR}")
logging.info(f"\t\tNumber of joblib feat files found to aggregate: {len(FEAT_FILES)}")
logging.info("\tState of filepaths in paths.py at time of run:")
logging.info(f"\t\tLABELS_FILE: {LABELS_FILE}")
logging.info(f"\t\tMRIS_DIR: {MRIS_DIR}")
logging.info(f"\t\tSEGS_DIR: {SEGS_DIR}")

all_feats = []
for f in tqdm(
    FEAT_FILES, desc="Aggregating pyrad feats...", total=len(FEAT_FILES), ncols=120
):
    # read in file
    result = joblib.load(f)
    feats = {}

    # parse file metadata
    f_metadata = f.name.replace(".joblib", "").replace("T1_POST", "T1POST").split("_")
    for md in f_metadata:

        md_name, md_val = md.split("-")
        feats[md_name] = md_val

    # parse file data
    for k, v in result.items():
        metadata, feat_class, feat_name = k.split("_")
        if metadata == "original":
            feats[f"{feat_class}_{feat_name}"] = v.item()

    all_feats.append(feats)

# collate
df_long = pd.DataFrame(all_feats)

# pivot to wide format
df_wide = df_long.pivot(index="subject", columns=["pulse", "seg"])
df_wide.columns = [f"{pulse}-{seg}-{feat}" for (feat, pulse, seg) in df_wide.columns]

# save
df_wide.to_csv(OUTPUT_FP)

# finish logging
logging.info(f"Features csv with shape {df_wide.shape} saved to {OUTPUT_FP}")
logging.info(f"END OF LOG ENTRY")
logging.info(f"<>" * 40)
