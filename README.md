# Computer Vision 3D Reconstruction

Python OpenGL 3.3

Order of execution:

1. calibration.py (only if intrinsics.avi is available, camera_params.pickle already contains the calibration from assignment 2)
2. annotation.py
3. background.py
4. data_processing.py
5. voxel.py
6. voxel_postprocessing.py
7. clustering.py
8. colour_model_offline.py
9. colour_model_online.py
10. tracking.py (change pickle file path for either online/offline tracking)
11. Change the pickle file path in assignment.py for the correct voxel model (see below)
12. executable.py

Voxel model options:

1. voxels_intersection.pickle: either with random colours or true
2. voxels_clusters.pickle: offline voxel clusters, passing centers onto next one
3. voxels_clusters_online.pickle: online clusters, random initialised centers, cluster assignment based on colour model

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
