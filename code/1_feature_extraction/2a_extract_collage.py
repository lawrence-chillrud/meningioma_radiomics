"""
File: 2a_extract_collage.py
Date: 10/07/2025
Author: Lawrence Chillrud <chili@u.northwestern.edu>

Extracts CoLlAGe features (https://collageradiomics.readthedocs.io/en/latest/) on
preprocessed MRI images for every annotation available in segmentation mask.
"""

import logging
from code.utils.get_cohort import get_cohort
from code.utils.paths import COLLAGE_DIR, LABELS_FILE, MRIS_DIR, SEGS_DIR
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

from tqdm import tqdm


def worker_func(
    subject_number, pulse_sequence, segmentation_label, bin_size, window_size
):
    # TODO
    raise NotImplementedError


def construct_jobs_list():
    subjects, _, _ = get_cohort(
        labels_file=LABELS_FILE, mris_dir=MRIS_DIR, segs_dir=SEGS_DIR
    )


def main():
    # Setup logging
    logging.basicConfig(
        filename="parallel_errors.log",
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Example input combinations
    subject_numbers = [1, 2, 3]
    pulse_sequences = ["T1", "T2"]
    segmentation_labels = ["GM", "WM"]
    bin_sizes = [10, 20]  # 20 will trigger exception
    window_sizes = [5, 10]

    args_list = list(
        product(
            subject_numbers,
            pulse_sequences,
            segmentation_labels,
            bin_sizes,
            window_sizes,
        )
    )

    errors = 0
    results = []

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker_func, *args): args for args in args_list}

        # tqdm updates as tasks complete
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing"
        ):
            args = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Log the error to file
                logging.error("Task %s failed with exception: %s", args, e)

    print("All tasks completed.")
    print(f"Successful results: {len(results)}")
    print(f"Failed tasks logged to parallel_errors.log")


if __name__ == "__main__":
    main()
