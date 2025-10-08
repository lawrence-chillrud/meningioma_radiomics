from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp


def read_ndarray(fname, original_shape=(189, 233, 197, 13, 2)):
    """
    Reads a given fname which is expected to be a scipy.sparse 2d matrix, reshapes it into the original collage
    feature shape it was expected to be saved from, and replaces 0s as NaNs. If a file with the same name as fname
    but with '_true_zero_locations.npy' exists, it will replace the NaNs at those locations with 0s, since that file
    indicates that the original array had 0s at those locations. Returns as a numpy ndarray of original_shape.
    """
    # Load scipy sparse 2D matrix and reshape it to the original collage feature shape
    dense_array = joblib.load(fname).toarray().reshape(original_shape)

    # Replace 0s with NaNs, since originally, NaNs were replaced with 0s in order to save as a sparse matrix
    dense_array[dense_array == 0] = np.nan

    # If the true zero locations file exists, replace the NaNs at those locations with 0s
    if Path(f"{fname}_true_zero_locations.npy").exists():
        true_zero_locations = np.load(f"{fname}_true_zero_locations.npy")
        for loc in true_zero_locations:
            dense_array[tuple(loc)] = 0

    return dense_array


def write_ndarray(arr, fname, check=False):
    """
    Writes a given numpy ndarray to a given filename as a sparse 2d matrix. NaN values are saved as 0s. Original locations of 0s saved to accompanying .npy file.
    """
    # save the locations of the true zeros (if they exist) so we can use them later...
    true_zero_locations = np.stack(np.where(arr == 0)).T
    if len(true_zero_locations):
        np.save(f"{fname}_true_zero_locations.npy", true_zero_locations)

    # copy array and replace NaNs with 0s, since scipy.sparse can't handle nans
    arr_zeros = arr.copy()
    arr_zeros[np.isnan(arr_zeros)] = 0

    # reshape the array to be a 2D matrix, since scipy.sparse can't handle more than 2D
    arr_zeros_reshaped = arr_zeros.reshape(arr.shape[0], -1)

    # convert to sparse matrix
    sparse_matrix = sp.csr_matrix(arr_zeros_reshaped)

    # save to disk
    joblib.dump(sparse_matrix, fname)

    # check to make sure everything worked smoothly
    if check:
        assert np.allclose(
            read_ndarray(fname), arr, equal_nan=True
        ), "Arrays are not equal"
