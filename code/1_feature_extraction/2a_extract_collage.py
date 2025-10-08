"""
Lawrence Chillrud <chili@u.northwestern.edu>

Extracts CoLlAGe features on our meningioma cohort (w/parallelization).

CoLlAGe docs: https://collageradiomics.readthedocs.io/en/latest/
"""

import logging
from code.utils import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from multiprocessing import cpu_count
from pathlib import Path

import collageradiomics
from ants import image_read
from tqdm import tqdm

# --- USER DEFINED GLOBAL VARS ---
TIMESTAMP = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
OUTPUT_DIR = COLLAGE_DIR / "a_raw_collage" / TIMESTAMP
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGFILE = OUTPUT_DIR / "logfile.txt"
HARALICK_WINDOW_SIZES = [3, 5]
BIN_SIZES = [32, 64]
SVD_RADIUS = 5


def run_collage(subject, pulse, seg, win, bin):
    """
    Extracts CoLlAGe features on provided subject, pulse, seg combination
    w/given CoLlAGe win & bin sizes. Saves to disk.
    """
    output_filepath = (
        OUTPUT_DIR
        / f"subject-{subject}_pulse-{pulse}_seg-{seg}_win-{win}_bin-{bin}.joblib"
    )

    # Read in MRI
    mri_path = get_mris(subject, pulses=pulse)[pulse]
    mri = image_read(str(mri_path), reorient="IAL").numpy()

    # Read in segmentation mask
    seg_mask = get_segs(subject, rois=seg)[seg]

    # Extract collage features
    collage = collageradiomics.Collage(
        mri,
        seg_mask,
        svd_radius=SVD_RADIUS,
        haralick_window_size=win,
        num_unique_angles=bin,
    )
    collage_features = collage.execute()

    # Save features to disk
    write_ndarray(collage_features, output_filepath)


def construct_jobs():
    """
    Enumerate all available combinations of subjects, pulse seqs, segmentation labels,
    window sizes, and bin sizes to later submit to workers.
    """
    subjects, _, _ = get_cohort(
        labels_file=LABELS_FILE, mris_dir=MRIS_DIR, segs_dir=SEGS_DIR
    )
    jobs = []
    for s in tqdm(
        subjects,
        desc="Step 1/2: Enumerating all CoLlAGe jobs needed",
        total=len(subjects),
    ):
        mris = list(get_mris(s).keys())
        segs = list(get_segs(s).keys())
        jobs.extend(list(product([s], mris, segs, HARALICK_WINDOW_SIZES, BIN_SIZES)))
    return jobs


def main():
    # Console output
    print("-" * 80)
    print(f"⏳ Running {Path(__file__).name}")
    print(f"📜 Logs will be saved to: {LOGFILE}")

    # Setup logfile
    logging.basicConfig(
        filename=LOGFILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Construct list of all jobs needed to be run (list of args to pass to wor)
    jobs_list = construct_jobs()
    N = len(jobs_list)

    # Start logfile
    logging.info(f"<>" * 40)
    logging.info(f"Log file for {Path(__file__).name} run at {TIMESTAMP}")
    logging.info(f"LABELS_FILE: {LABELS_FILE}")
    logging.info(f"MRIS_DIR: {MRIS_DIR}")
    logging.info(f"SEGS_DIR: {SEGS_DIR}")
    logging.info(f"CoLlAGe window sizes: {HARALICK_WINDOW_SIZES}")
    logging.info(f"CoLlAGe bin sizes: {BIN_SIZES}")
    logging.info(f"CoLlAGe SVD radius (fixed): {SVD_RADIUS}")
    logging.info(f"Number of jobs to run: {N}")
    logging.info(f"Results saved to output directory: {OUTPUT_DIR}")
    logging.info(f"-" * 20)
    successes = 0
    errors = 0

    # Parallel loop
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = {executor.submit(run_collage, *args): args for args in jobs_list}

        for future in tqdm(
            as_completed(futures), total=N, desc="Step 2/2: Completed CoLlAGe job"
        ):
            args = futures[future]
            try:
                _ = future.result()
                logging.info(
                    f"✅ Job success for args (subject={args[0]}, pulse={args[1]}, seg={args[2]}, win={args[3]}, bin={args[4]})"
                )
                successes += 1
            except Exception as e:
                logging.error(
                    f"❌ Job failed for args (subject={args[0]}, pulse={args[1]}, seg={args[2]}, win={args[3]}, bin={args[4]}) with error: {e}"
                )
                errors += 1

    # End logfile
    logging.info(f"-" * 20)
    logging.info(f"All {N} jobs completed.")
    logging.info(f"Successful jobs: {successes}")
    logging.info(f"Jobs ending in error: {errors}")
    logging.info(f"<>" * 40)

    # End console output
    print(f"✅ Finished running {Path(__file__).name}")
    print(f"📂 Logs have been saved to: {LOGFILE}")


if __name__ == "__main__":
    main()
