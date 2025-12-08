"""
Lawrence Chillrud <chili@u.northwestern.edu>

Extracts PyRadiomics features on our meningioma cohort (w/parallelization).

PyRadiomics docs: https://pyradiomics.readthedocs.io/en/latest/
"""
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
# from ants import image_read
from itertools import product
from tqdm import tqdm
import radiomics
from radiomics import featureextractor
from src.utils import *
import joblib

# --- USER DEFINED GLOBAL VARS ---
TIMESTAMP = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
OUTPUT_DIR = PYRAD_DIR / "a_raw_pyradiomics" / TIMESTAMP
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGFILE = OUTPUT_DIR / "logfile.txt"
MAX_WORKERS = cpu_count()
radiomics.setVerbosity(level=60) # logging.INFO or level=60
EXTRACTOR = featureextractor.RadiomicsFeatureExtractor(correctMask=True)
EXTRACTOR.enableAllFeatures()


def run_pyrad(subject, pulse, seg):
    """
    Extracts PyRadiomics features on provided subject, pulse, seg combination.
    Saves to disk.
    """
    output_filepath = (
        OUTPUT_DIR
        / f"subject-{subject}_pulse-{pulse}_seg-{seg}.joblib"
    )

    # Read in MRI
    mri_path = get_mris(subject, pulses=pulse)[pulse]
    # mri = image_read(str(mri_path), reorient="IAL").numpy()

    # Read in segmentation mask
    seg_mask = get_segs(subject, rois=seg, with_sitk=True)[seg]

    # Extract PyRadiomics features
    result = EXTRACTOR.execute(str(mri_path), seg_mask, label=1)

    # Save features to disk
    joblib.dump(result, output_filepath)

def construct_jobs():
    """
    Enumerate all available combinations of subjects, pulse seqs & segmentation labels
    to later submit to workers.
    """
    subjects, _, _ = get_cohort(
        labels_file=LABELS_FILE, mris_dir=MRIS_DIR, segs_dir=SEGS_DIR
    )
    jobs = []
    for s in tqdm(
        subjects,
        desc="Step 1/3: Enumerating all possible PyRadiomics jobs",
        total=len(subjects),
        ncols=120,
    ):
        mris = list(get_mris(s).keys())
        segs = list(get_segs(s).keys())
        jobs.extend(list(product([s], mris, segs)))

    jobs_left = []
    for j in tqdm(
        jobs,
        desc="Step 2/3: Checking which jobs still need doing",
        total=len(jobs),
        ncols=120,
    ):
        output_filepath = (
            OUTPUT_DIR
            / f"subject-{j[0]}_pulse-{j[1]}_seg-{j[2]}.joblib"
        )
        if not output_filepath.exists():
            jobs_left.append(j)

    return jobs_left


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
    logging.info(f"Number of jobs to run: {N}")
    logging.info(f"Results saved to output directory: {OUTPUT_DIR}")
    logging.info(f"MAX_WORKERS: {MAX_WORKERS}")
    logging.info(f"-" * 20)
    successes = 0
    errors = 0

    # Parallel loop
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_pyrad, *args): args for args in jobs_list}

        for future in tqdm(
            as_completed(futures),
            total=N,
            desc="Step 3/3: Completed PyRadiomics job",
            ncols=120,
        ):
            args = futures[future]
            try:
                _ = future.result()
                logging.info(
                    f"✅ Job success for args (subject={args[0]}, pulse={args[1]}, seg={args[2]})"
                )
                successes += 1
            except Exception as e:
                logging.error(
                    f"❌ Job failed for args (subject={args[0]}, pulse={args[1]}, seg={args[2]}) with error: {e}"
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
