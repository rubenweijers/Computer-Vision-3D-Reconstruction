import matplotlib.pyplot as plt
import numpy as np

from data_processing import load_pickle


def plot_clusters(voxels, bounds: dict):
    """Plot the X and Y coordinates of the voxels in a scatter plot."""
    plt.scatter(voxels[:, 0], voxels[:, 1], alpha=0.05)  # Use alpha to show density
    plt.gca().set_aspect("equal")

    plt.xlim(bounds["x_lowerbound"], bounds["x_upperbound"])
    plt.ylim(bounds["y_lowerbound"], bounds["y_upperbound"])

    plt.xlabel("Side view (mm)")
    plt.ylabel("Front view (mm)")
    plt.show()


if __name__ == "__main__":
    data = load_pickle("./data/voxels_intersection.pickle")
    voxels = data["voxels"]
    bounds = data["bounds"]

    voxels = np.array(voxels)
    voxels = voxels * bounds["stepsize"]  # Scale the voxel by step size
    voxels = voxels[:, [0, 2]]  # Select only X and Y, original order: X, Z, Y

    plot_clusters(voxels, bounds)
