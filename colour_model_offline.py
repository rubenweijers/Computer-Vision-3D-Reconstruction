import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm import trange

from data_processing import load_pickle, save_pickle

colour_map = {0: [0, 1, 1], 1: [1, 0, 1], 2: [1, 1, 0], 3: [0, 0, 0]}  # Map cluster to colour
colour_map.update({tuple(v): k for k, v in colour_map.items()})  # Add reverse mapping


def get_colour_subset(frame_voxels, frame_colours, frame_labels, cluster: int, hip_height: int = 850, shoulder_height: int = 1500):
    """Get all colours from a specific cluster, and filter based on z axis (height)"""
    idx = np.where((frame_labels == cluster))  # All idx from the same cluster
    colour_subset = frame_colours[idx]  # Filter based on cluster

    # All where z axis is between 70-150 centimeters, order is x, z, y
    if hip_height is not None and shoulder_height is not None:  # Only filter if both are not None
        voxels_subset = frame_voxels[idx]  # Filter based on cluster

        idx_z = np.where((voxels_subset[:, 1] >= hip_height) & (voxels_subset[:, 1] <= shoulder_height))
        colour_subset = colour_subset[idx_z]
    return colour_subset


def convert_colour_space_of_list(colour_list: np.ndarray, colour_space: cv2.COLOR_RGB2LAB) -> np.ndarray:
    """Convert list of colours to LAB colour space. OpenCV requires 3D array to convert colour space"""
    colour_list = colour_list.astype(np.float32).reshape(-1, 1, 3)  # Reshape to 3D array
    colour_list = cv2.cvtColor(colour_list, colour_space)  # Convert to new colour space
    return colour_list.reshape(-1, 3)  # Reshape back to 2D array


def get_mean_gmm(colour_subset, n_gmm_clusters: int) -> np.ndarray:
    """Get mean of GMM, which is the colour model"""
    gmm = cv2.ml.EM_create()
    gmm.setClustersNumber(n_gmm_clusters)

    gmm.trainEM(colour_subset)  # Train on all colours of this cluster
    return gmm.getMeans()  # n_gmm_clusters number of colours


def get_mean_kmeans(colour_subset, n_kmeans_clusters: int) -> np.ndarray:
    """Get mean of KMeans, which is the colour model"""
    model = KMeans(n_clusters=n_kmeans_clusters).fit(colour_subset)
    return model.cluster_centers_


if __name__ == "__main__":
    data_voxels = load_pickle("./data/voxels_intersection.pickle")
    data_clusters = load_pickle("./data/voxels_clusters.pickle")
    n_clusters = 2  # Number of colours per person for the colour model
    use_lab_colour_space = True  # Convert to LAB colour space
    use_gmm = True  # Use GMM, otherwise KMeans

    voxels = data_voxels["voxels"]
    colours = data_voxels["colours"]
    bounds = data_voxels["bounds"]
    clusters = data_clusters["colours"]

    # Select first frame and cluster
    frame_voxels = np.array(voxels[0]) * bounds["stepsize"]  # Convert back to mm
    frame_colours = np.array(colours[0])
    frame_labels = np.array(clusters[0])
    frame_labels = np.array([colour_map[tuple(cluster)] for cluster in frame_labels])  # Replace colour with label, 0-3

    # Print shapes
    print("Voxels shape: ", frame_voxels.shape)
    print("Colours shape: ", frame_colours.shape)
    print("Clusters shape: ", frame_labels.shape)

    colour_models = []
    for cluster in trange(4):
        colour_subset = get_colour_subset(frame_voxels, frame_colours, frame_labels, cluster)

        if use_lab_colour_space:
           # Convert to LAB colour space, based on sciencedirect.com/science/article/pii/S1077314209000496
            colour_subset = convert_colour_space_of_list(colour_subset, colour_space=cv2.COLOR_RGB2LAB)

        if use_gmm:  # Make colour model with GMM
            colour_model = get_mean_gmm(colour_subset, n_clusters)
        else:  # Make colour model with KMeans
            colour_model = get_mean_kmeans(colour_subset, n_clusters)

        colour_models.append(colour_model)

    # Calculate distance between means of each colour model
    distance = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            distance[i][j] = np.linalg.norm(colour_models[i] - colour_models[j])
    distance = np.tril(distance)  # Remove upper triangle, since duplicate
    print(distance.round(2))

    # Convert to numpy array
    colour_models = np.array(colour_models)
    print(colour_models.shape)
    print(colour_models)

    # Save colour models to pickle
    data = {"colour_models": colour_models, "n_clusters": n_clusters,
            "use_lab_colour_space": use_lab_colour_space, "use_gmm": use_gmm}
    save_pickle("./data/colour_models.pickle", data)

    # Create image
    height = 300
    width = 600
    every_h = int(height / n_clusters)
    every_w = int(width / 4)
    img = np.zeros((height, width, 3), dtype=np.float32)
    for colour in range(n_clusters):
        for person in range(4):
            img[colour * every_h:(colour + 1) * every_h, person * every_w:(person + 1) * every_w] = colour_models[person][colour]

    # Convert to BGR
    if use_lab_colour_space:
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Show image
    cv2.imshow("Colour model", img)
    cv2.waitKey(0)
