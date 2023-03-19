import matplotlib.pyplot as plt
import numpy as np

from colour_model_offline import colour_map, get_colour_subset
from data_processing import load_pickle


def plot_cluster_centers(n_clusters=4, colours=None, all_cluster_centers=None, bounds=None, window_size: int = 3):
    for cluster in range(n_clusters):
        cluster_centers = all_cluster_centers[:, cluster, :]
        xs = cluster_centers[:, 0]
        ys = cluster_centers[:, 1]

        xs = np.convolve(xs, np.ones(window_size) / window_size, mode="same")  # Calculate floating average of cluster centers
        ys = np.convolve(ys, np.ones(window_size) / window_size, mode="same")  # Window size is hyperparameter

        plt.plot(xs, ys, color=colours[cluster], label=f"Cluster {cluster}", marker="o")

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
    window_size = 3
    mode = "offline"  # Either "online" or "offline"

    if mode == "online":
        data = load_pickle("./data/voxels_clusters_online.pickle")
        bounds = data["bounds"]
        all_voxels = data["voxels"]
        all_colours = data["colours"]

        all_cluster_centers = []
        for frame_voxels, frame_colours in zip(all_voxels, all_colours):
            frame_colours = np.array([colour_map[tuple(colour)] for colour in frame_colours])  # Convert to cluster number instead of RGB
            frame_voxels = np.array(frame_voxels) * bounds["stepsize"]  # Convert to mm
            frame_voxels = frame_voxels[:, [0, 2]]  # Only keep X and Y coords, order: X, Z, Y

            cluster_centers = []
            for cluster in range(4):
                # Seperate voxels into four clusters
                # Ignore hip and shoulder height since we want to track the whole person/cluster
                cluster_voxels = get_colour_subset(frame_voxels, frame_voxels, frame_colours,
                                                   cluster, hip_height=None, shoulder_height=None)
                cluster_center = np.mean(cluster_voxels, axis=0)  # Cluster center is determined as the average of all voxels in the cluster
                cluster_centers.append(cluster_center)
            all_cluster_centers.append(cluster_centers)
        all_cluster_centers = np.array(all_cluster_centers)

    elif mode == "offline":
        data = load_pickle("./data/voxels_clusters.pickle")
        bounds = data["bounds"]
        all_cluster_centers = np.array(data["cluster_centers"])

    # N x 4 x 2, where N is the number of frames, 4 clusters, and 2 is X and Y coords
    # print(all_cluster_centers.shape)

    plot_cluster_centers(n_clusters, colours, all_cluster_centers, bounds, window_size=window_size)
