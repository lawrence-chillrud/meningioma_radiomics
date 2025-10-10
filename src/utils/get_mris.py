from pathlib import Path
from typing import List, Union

from .paths import MRIS_DIR, PULSES


def get_mris(
    subject: Union[str, int],
    mris_dir: Path = MRIS_DIR,
    pulses: Union[str, List[str]] = PULSES,
):
    """
    Given a subject ID number, returns the filepath(s) of the MRI(s) for the specified pulse sequence(s) of interest.

    Parameters:
    -----------
    subject (str or int): The subject ID number.
    mris_dir (Path): The directory containing the MRIs.
    pulses (List[str]): The pulse sequence(s) of interest.

    Returns:
    --------
    (dict or None): A dictionary containing the filepath(s) of the MRI(s) for the specified pulse sequence(s) of interest.
    """
    active_dir = sorted(
        [p for p in (Path(mris_dir) / str(subject)).iterdir() if p.is_dir()]
    )[0]
    available_mris = [
        p for p in active_dir.rglob("*") if p.is_file() and ".nii" in p.name
    ]

    if isinstance(pulses, str):
        pulses = [pulses]

    mris = {}
    for p in pulses:
        for m in available_mris:
            if p in str(m):
                mris[p] = m
    return mris
