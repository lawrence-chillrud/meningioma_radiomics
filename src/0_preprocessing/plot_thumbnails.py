# %%
import os

os.chdir("../..")
os.getcwd()
from src.utils.paths import *
from ants import image_read
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from ipywidgets import interact

OUTPUT_DIR = PREPROC_DIR / "THUMBNAILS"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
dirs = [
    d
    for d in PREPROC_DIR.iterdir()
    if not d.name.startswith("0") and d.is_dir() and not d.name.startswith("8")
]


def make_thumbnail(f, output_dir=None, slice_num=None):
    im = image_read(str(f), reorient="IAR")
    im_numpy = im.numpy()
    if slice_num:
        plt.imshow(im.numpy()[slice_num, :, :], cmap="gray")
    else:
        num_axial_slices = im_numpy.shape[0]
        plt.imshow(im.numpy()[num_axial_slices // 2, :, :], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    if output_dir:
        output_fn = f.name.split(".nii")[0]
        output_fp = output_dir / f"{output_fn}.jpg"
        plt.savefig(output_fp, dpi=600)
    else:
        plt.show()
    plt.close()


# %%
for d in tqdm(dirs, total=len(dirs)):
    output_dir = OUTPUT_DIR / d.name
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [f for f in d.rglob("*.nii*")]
    for f in files:
        make_thumbnail(f, output_dir)


# %%
def explore_3D_array(arr: np.ndarray, cmap: str = "gray"):
    """
    Given a 3D array with shape (Z,X,Y) This function will create an interactive
    widget to check out all the 2D arrays with shape (X,Y) inside the 3D array.
    The purpose of this function to visual inspect the 2D arrays in the image.

    Args:
      arr : 3D array with shape (Z,X,Y) that represents the volume of a MRI image
      cmap : Which color map use to plot the slices in matplotlib.pyplot
    """

    def fn(SLICE):
        plt.figure(figsize=(7, 7))
        plt.imshow(arr[SLICE, :, :], cmap=cmap)

    interact(fn, SLICE=(0, arr.shape[0] - 1))


# %%
