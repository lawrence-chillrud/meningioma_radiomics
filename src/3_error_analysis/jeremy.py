# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from src.utils import MRIS_DIR, METADATA_FILE
import json
from functools import reduce

def load_json(fp):
    with open(fp, 'r') as file:
        data = json.load(file)
    
    return data

def load_json_metadata_for_subs(fields_oi, subs_list=None):
    if subs_list is None:
        subjects_of_interest = [f for f in MRIS_DIR.iterdir() if f.is_dir()]
    else:
        subjects_of_interest = [f for f in MRIS_DIR.iterdir() if f.is_dir() and int(f.name) in subs_list]

    running_jsons = []
    running_keys = []
    for subject in subjects_of_interest:
        available_jsons = [f for f in subject.rglob("*.json")]
        for jf in available_jsons:
            data = load_json(jf)
            data['Subject Number'] = subject.name
            data['Pulse Sequence'] = jf.name.split('-')[-1].replace('.json', '')
            fn_part1 = jf.name.split('-')[0]
            session_parts = fn_part1.split('_')
            session_name = session_parts[1].lower() if len(session_parts) == 3 else session_parts[1].lower() + "_" + session_parts[2].lower()
            data['Session'] = session_name
            keys = list(data.keys())
            running_jsons.append(data)
            running_keys.append(keys)

    # common_keys = sorted(list(reduce(set.intersection, [set(item) for item in running_keys])))

    subset_dicts = []
    for d in running_jsons:
        subset_dicts.append({k: d[k] if k in d else None for k in fields_oi})

    metadata_df = pd.DataFrame(subset_dicts)

    return metadata_df

# %%
fields_of_interest = [
    'Subject Number', 'Session', 'Pulse Sequence', 'MagneticFieldStrength', 'InstitutionName',
    'ProtocolName', 'SliceThickness', 'SpacingBetweenSlices', 'EchoTime', 'RepetitionTime',
    'FlipAngle', 'PhaseEncodingSteps', 'AcquisitionMatrixPE', 'ReconMatrixPE'
]
# train_only_subs = [13, 31, 34, 35, 38, 42, 48, 52, 75, 87, 91, 92, 94, 97, 104, 109, 110, 115]
# frequent_val_fliers = [28, 43, 68, 98, 111, 117]

all_subs = load_json_metadata_for_subs(fields_of_interest)
metadata_df = pd.read_csv(METADATA_FILE).drop(columns=["Session"])

all_subs['Subject Number'] = all_subs['Subject Number'].astype(int)
metadata_df['Subject Number'] = metadata_df['Subject Number'].astype(int)

main_metadata_df = all_subs.join(metadata_df.set_index('Subject Number'), on=['Subject Number'], how='left').sort_values(by=['Subject Number', 'Pulse Sequence'])
# %%
main_metadata_df.to_csv('meningioma_cohort_imaging_metadata.csv', index=False)

# %%
from src.utils import get_feats, PYRAD_FILE

methyl_X, methyl_y, methyl_subs = get_feats(
    prediction_task="MethylationSubgroup",
    features_path=PYRAD_FILE,
    drop_extra_shape_feats=False,
    remove_correlated_feats=None,
    drop_constant_feats=False
)

chr1p_X, chr1p_y, chr1p_subs = get_feats(
    prediction_task="Chr1p",
    features_path=PYRAD_FILE,
    drop_extra_shape_feats=False,
    remove_correlated_feats=None,
    drop_constant_feats=False
)

chr22q_X, chr22q_y, chr22q_subs = get_feats(
    prediction_task="Chr22q",
    features_path=PYRAD_FILE,
    drop_extra_shape_feats=False,
    remove_correlated_feats=None,
    drop_constant_feats=False
)

# %%
train_only[fields_of_interest]

# %%
frequent_fliers[fields_of_interest]
# %%
metadata_df = pd.read_csv(METADATA_FILE)
metadata_df[metadata_df['Subject Number'].isin(frequent_val_fliers)]

# %%
metadata_df[metadata_df['Subject Number'].isin(train_only_subs)]

# %%
