# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd

from src.utils import COLLAGE_DIR

# %%
df = pd.read_csv(
    COLLAGE_DIR
    / "b_aggregated_collage"
    / "10-10-2025_19-17-50"
    / "features-collage_win-5_bin-64.csv"
)
# %%
wide_df = df.set_index(
    ["subject", "win", "bin", "pulse", "seg", "collage_feat", "collage_angle"]
).unstack(["pulse", "seg", "collage_feat", "collage_angle"])

wide_df.columns = [
    f"{pulse}_{seg}_{feat}_{angle}_{col}"
    for col, pulse, seg, feat, angle in wide_df.columns.to_flat_index()
]

wide_df = wide_df.reset_index()

wide_df
