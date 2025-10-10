# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# from .get_segs import get_segs
# from .get_mris import get_mris
import matplotlib.pyplot as plt
import numpy as np
from ants import image_read

from src.utils import get_mris, get_segs


def make_thumbnail(subject, pulse="T1_POST", cslice=None):
    # Read in segmentation, get max cancerous slice
    seg = get_segs(subject, rois=22)[22]
    cancerous_pixels_per_slice = np.sum(seg, axis=(1, 2))
    if cslice is None:
        cslice = np.argmax(cancerous_pixels_per_slice)

    print(cslice)
    # Read in mri's max cancerous slice
    mri_slice = image_read(
        str(get_mris(subject, pulses=pulse)[pulse]), reorient="IAL"
    ).numpy()[cslice, :, :]

    # Plot
    plt.imshow(mri_slice, cmap="gray")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.show()


# %%
