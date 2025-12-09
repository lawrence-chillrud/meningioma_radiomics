# Feature extraction

Two distinct radiomics feature extraction toolkits were employed for feature extraction: [PyRadiomics](https://pyradiomics.readthedocs.io/en/latest/#) version 3.0.1 with default parameters, and [CoLlAGe](https://collageradiomics.readthedocs.io/en/latest/) version 0.3.8 using varying parameter settings. In both cases, features were extracted relative to every combination of four input pulse sequences and ten input segmentation masks, yielding 40 instances of each feature per patient:
- The four pulse sequences consisted of: __T1 POST, FLAIR, DWI, and ADC__.
- Input segmentations included ten distinct tumor components or combinations of components: __whole tumor, necrotic tumor, enhancing tumor, edema, susceptibility, restricted diffusion, enhancing with necrotic tumor, enhancing with susceptibility, enhancing with restricted diffusion, and enhancing with susceptibility and restricted diffusion.__

## 1. PyRadiomics

In the case of [PyRadiomics](https://pyradiomics.readthedocs.io/en/latest/#), the following default set of features were extracted:

- first-order grayscale features (n=18),
- 3D shape features (n=14),
- gray level co-occurrence matrix features (n=24),
- gray level run length matrix features (n=16),
- gray level size zone matrix features (n=16),
- neighboring gray tone difference matrix features (n=5),
- and gray level dependence matrix features (n=14).

Extracting these 107 features for all combinations of the four pulse sequences and ten segmentation annotations yielded 4,280 PyRadiomics features per patient.

## 2. CoLlAGe

In the case of [CoLlAGe](https://collageradiomics.readthedocs.io/en/latest/), we extracted the default set of 13 Haralick 3D texture maps in both the primary and secondary angles before summarizing each of these resulting 26 texture maps using 19 sample statistics, yielding 494 continuous features. Extracting these 494 features for all combinations of the four pulse sequences and ten segmentation annotations yielded 19,760 CoLlAGe features per patient.

__The sensitivity of CoLlAGe features to its two key parameters, bin size of the entropy histogram (v) and neighborhood size (N) for computing localized orientations, was evaluated by extracting CoLlAGe features for all combinations of v∈{16, 32, 48, 64} and N∈{3, 5, 7, 9}, yielding sixteen distinct CoLlAGe feature sets in all.__
