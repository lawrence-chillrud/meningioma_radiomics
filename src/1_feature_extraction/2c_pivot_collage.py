# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd

from src.utils import COLLAGE_DIR

# %%
collage_files = [f for f in COLLAGE_DIR.rglob("*.csv")]

for f in collage_files:

    df = pd.read_csv(f).rename(columns={"subject": "Subject Number"})

    win_size = df["win"][0]
    bin_size = df["bin"][0]

    wide_df = df.set_index(
        ["Subject Number", "win", "bin", "pulse", "seg", "collage_feat", "collage_angle"]
    ).unstack(["pulse", "seg", "collage_feat", "collage_angle"])

    wide_df.columns = [
        f"{pulse}_{seg}_{feat}_{angle}_{col}"
        for col, pulse, seg, feat, angle in wide_df.columns.to_flat_index()
    ]

    wide_df = wide_df.reset_index().drop(columns=["win", "bin"])

    new_filepath = f.parent / ("wide-" + f.name)
    wide_df.to_csv(new_filepath, index=False)
    
# %%
