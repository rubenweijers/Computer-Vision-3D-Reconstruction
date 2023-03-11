import pickle

import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm.contrib import tzip

from clustering import plot_clusters
from colour_model_offline import (colour_map, convert_colour_space_of_list,
                                  get_colour_subset, get_mean_gmm,
                                  get_mean_kmeans)
from data_processing import load_pickle, save_pickle

if __name__ == "__main__":
    data = load_pickle("./data/voxels_intersection.pickle")
    all_voxels = data["voxels"]
    all_colours = data["colours"]
    bounds = data["bounds"]

    data_colour_model = load_pickle("./data/colour_models.pickle")
    colour_models = data_colour_model["colour_models"]

    all_labels = []
    all_cluster_centers = []
    for frame_voxels, frame_colours in tzip(all_voxels, all_colours):
        frame_voxels = np.array(frame_voxels)
        frame_voxels = frame_voxels * bounds["stepsize"]  # Scale the voxel by step size
        frame_voxels_xy = frame_voxels[:, [0, 2]]
        frame_colours = np.array(frame_colours)

        # Cluster into four groups with sklearn, random cluster centers
        model = KMeans(n_clusters=4, n_init=1)
        model.fit(frame_voxels_xy)
        frame_labels = model.predict(frame_voxels_xy)  # Get labels for each point, 0-3

        for cluster in range(4):
            # Get all colours from this cluster
            colour_subset = get_colour_subset(frame_voxels, frame_colours, frame_labels, cluster)

            if data_colour_model["use_lab_colour_space"]:
                colour_subset = convert_colour_space_of_list(colour_subset.copy(), cv2.COLOR_RGB2LAB)

            if data_colour_model["use_gmm"]:
                cluster_centers = get_mean_gmm(colour_subset, data_colour_model["n_clusters"])
            else:
                cluster_centers = get_mean_kmeans(colour_subset, data_colour_model["n_clusters"])

        # Save the cluster centers to analyse with offline clustering
        all_cluster_centers.append(model.cluster_centers_)

        # Convert labels to RGB
        frame_labels = [colour_map[label] for label in frame_labels]
        all_labels.append(frame_labels)

        # Plot the clusters
        plot_clusters(frame_voxels_xy, bounds, frame_labels)

    # Save to pickle
    data = {"voxels": data["voxels"], "bounds": data["bounds"], "colours": all_labels, "cluster_centers": all_cluster_centers}
    save_pickle("./data/voxels_clusters_online.pickle", data)
