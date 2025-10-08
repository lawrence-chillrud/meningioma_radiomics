from ants import image_read

from . import lsdir


def get_mris(subject_mri_dir, pulse_sequences):
    session = subject_mri_dir.split("/")[-1]
    mris = {}
    scans = lsdir(subject_mri_dir)
    for ps in pulse_sequences:
        for scan in scans:
            if scan.lower().endswith(ps.lower()):
                mris[ps] = image_read(
                    f"{subject_mri_dir}/{scan}/{session}_{scan}.nii.gz", reorient="IAL"
                ).numpy()
                break
    return mris
