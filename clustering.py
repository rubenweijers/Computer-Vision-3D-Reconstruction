import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from data_processing import load_pickle


def plot_clusters(voxels, bounds: dict, cluster_labels=None):
    """Plot the X and Y coordinates of the voxels in a scatter plot."""
    plt.scatter(voxels[:, 0], voxels[:, 1], alpha=0.05, c=cluster_labels)  # Use alpha to show density
    plt.gca().set_aspect("equal")

    plt.xlim(bounds["x_lowerbound"], bounds["x_upperbound"])
    plt.ylim(bounds["y_lowerbound"], bounds["y_upperbound"])

    plt.xlabel("Side view (mm)")
    plt.ylabel("Front view (mm)")
    plt.show()


if __name__ == "__main__":
    # Colour dict based on CYMK converted to RGB
    colours = {0: (0, 1, 1), 1: (1, 0, 1), 2: (1, 1, 0), 3: (0, 0, 0)}

    data = load_pickle("./data/voxels_intersection.pickle")
    all_voxels = data["voxels"]
    bounds = data["bounds"]

    all_labels = []
    cluster_centers = "k-means++"  # Use the default k-means++ to initialise the cluster centers
    all_cluster_centers = []
    for frame_voxels in all_voxels:
        frame_voxels = np.array(frame_voxels)
        frame_voxels = frame_voxels * bounds["stepsize"]  # Scale the voxel by step size
        frame_voxels = frame_voxels[:, [0, 2]]  # Select only X and Y, original order: X, Z, Y
        # plot_clusters(frame_voxels, bounds)

        # Cluster into four groups with sklearn
        model = KMeans(n_clusters=4, init=cluster_centers, n_init=1)
        model.fit(frame_voxels)
        labels = model.predict(frame_voxels)

        cluster_centers = model.cluster_centers_  # Use the cluster centers as the initialisation for the next frame
        all_cluster_centers.append(cluster_centers)

        # Convert labels to RGB
        labels = [colours[label] for label in labels]
        all_labels.append(labels)

        # Plot the clusters
        plot_clusters(frame_voxels, bounds, labels)

    # Save to pickle
    data = {"voxels": data["voxels"], "bounds": data["bounds"], "colours": all_labels, "cluster_centers": all_cluster_centers}
    with open("./data/voxels_clusters.pickle", "wb") as fp:
        pickle.dump(data, fp)
