# Computer Vision 3D Reconstruction

Python OpenGL 3.3

Order of execution:

1. `calibration.py` (Only if `intrinsics.avi` is available, `camera_params.pickle` already contains the calibration from assignment 2)
2. `annotation.py`
3. `background.py`
4. `data_processing.py`
5. `voxel.py` (Change`n_frames`and`every_nth` for number of frames in total and stepsize of frames to skip)
6. `voxel_postprocessing.py`
7. `clustering.py`(Change number of clusters of the colour model, default: `2`. Additional options to use either GMM or KMeans clustering and use either LAB or RGB colour space.)
8. `colour_model_offline.py`
9. `colour_model_online.py`
10. `tracking.py` (Change `mode` variable for either online/offline tracking)
11. `assignment.py` (Change the `mode` variable for either online/offline/basic voxel model + colouring)
12. `executable.py`

Voxel model options:

1. `voxels_intersection.pickle`: either with random colours or true, dependent on the `colourise` variable selected in `voxel.py`
2. `voxels_clusters.pickle`: offline voxel clusters, passing voxel cluster centers onto next frame
3. `voxels_clusters_online.pickle`: online clusters, random initialised centers for every frame, cluster assignment based on colour model

## Installation

Tested with Python 3.9.0. You can see all the libraries and versions in requirements.txt.

Install following packages via pip:

1. PyOpenGL: `pip install PyOpenGL PyOpenGL_accelerate`
2. GLFW: `pip install glfw`
3. PyGLM: `pip install PyGLM`
4. numpy: `pip install numpy`
5. Pillow: `pip install Pillow`

If you have errors after installation you will do following:

1. `pip uninstall PyOpenGL PyOpenGL_accelerate`
2. Download and install PyOpenGL and PyOpenGL_accelerate from <https://www.lfd.uci.edu/~gohlke/pythonlibs/>

## Control

```text
G - to visualize new voxel array
ESC - to exit the program
```

## Execution

Open your terminal or CMD and call `python executable.py` or `python3 executable.py`

## Thanks

- stanfortonski - <https://github.com/stanfortonski> - for providing such a great codebase.
