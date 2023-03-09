import matplotlib.pyplot as plt
import numpy as np

from data_processing import load_pickle


def plot_cluster_centers(n_clusters=4, colours=None, all_cluster_centers=None, bounds=None):
    for cluster in range(n_clusters):
        cluster_centers = all_cluster_centers[:, cluster, :]
        plt.plot(cluster_centers[:, 0], cluster_centers[:, 1], color=colours[cluster], label=f"Cluster {cluster}", marker="o")

    # Indicate start point of each cluster
    for cluster in range(n_clusters):
        cluster_centers = all_cluster_centers[:, cluster, :]
        plt.plot(cluster_centers[0, 0], cluster_centers[0, 1], color="red", marker="X", markersize=10)

    plt.gca().set_aspect("equal")
    plt.xlim(bounds["x_lowerbound"], bounds["x_upperbound"])
    plt.ylim(bounds["y_lowerbound"], bounds["y_upperbound"])
    plt.xlabel("Side view (mm)")
    plt.ylabel("Front view (mm)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    colours = {0: (0, 1, 1), 1: (1, 0, 1), 2: (1, 1, 0), 3: (0, 0, 0)}
    n_clusters = 4

    data = load_pickle("./data/voxels_clusters.pickle")
    bounds = data["bounds"]
    all_cluster_centers = data["cluster_centers"]
    all_cluster_centers = np.array(all_cluster_centers)

    plot_cluster_centers(n_clusters, colours, all_cluster_centers, bounds)
