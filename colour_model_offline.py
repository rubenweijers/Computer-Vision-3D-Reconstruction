import cv2
import numpy as np

from data_processing import load_pickle

if __name__ == "__main__":
    data_voxels = load_pickle("./data/voxels_intersection.pickle")
    data_clusters = load_pickle("./data/voxels_clusters.pickle")
    colour_map = {0: [0, 1, 1], 1: [1, 0, 1], 2: [1, 1, 0], 3: [0, 0, 0]}
    n_gmm_clusters = 3  # Number of colours per person for the colour model
    use_lab_colour_space = False  # Convert to LAB colour space

    voxels = data_voxels["voxels"]
    colours = data_voxels["colours"]
    clusters = data_clusters["colours"]

    # Select first frame and cluster
    frame_voxels = np.array(voxels[0]) * 30 / 10  # Convert back to mm, then to cm
    frame_colours = np.array(colours[0])
    frame_clusters = np.array(clusters[0])

    # Print shapes
    print("Voxels shape: ", frame_voxels.shape)
    print("Colours shape: ", frame_colours.shape)
    print("Clusters shape: ", frame_clusters.shape)

    colour_models = []
    for cluster in colour_map:
        # All where z axis is between 70-150 centimeters, order is x, z, y
        idx_z = np.where((frame_voxels[:, 1] >= 70) & (frame_voxels[:, 1] <= 150))
        idx = np.where((frame_clusters == colour_map[cluster]).all(axis=1))  # All idx from the same cluster
        colour_subset = frame_colours[np.intersect1d(idx, idx_z)]  # Filter based on both conditions

        if use_lab_colour_space:
            # Convert to LAB colour space, based on sciencedirect.com/science/article/pii/S1077314209000496
            colour_subset = colour_subset.astype(np.float32).reshape(-1, 1, 3)  # Reshape to 3D array to convert to LAB
            colour_subset = cv2.cvtColor(colour_subset, cv2.COLOR_RGB2LAB)  # Convert to LAB
            colour_subset = colour_subset.reshape(-1, 3)  # Reshape back to 2D array

        # Make colour model with GMM of each of the four clusters
        gmm = cv2.ml.EM_create()
        gmm.setClustersNumber(n_gmm_clusters)

        gmm.trainEM(colour_subset)  # Train on all colours of this cluster
        colour_model = gmm.getMeans()  # n_gmm_clusters number of colours
        colour_models.append(colour_model)

    # Calculate distance between means of each colour model
    for i in range(len(colour_models)):
        for j in range(len(colour_models)):
            if i != j:
                print(f"Distance between colour {i} and colour {j}: ", np.linalg.norm(colour_models[i] - colour_models[j]).round(2))

    # Convert to numpy array
    colour_models = np.array(colour_models)
    print(colour_models.shape)
    print(colour_models)

    # Create image
    img = np.zeros((300, 600, 3), dtype=np.float32)
    for i in range(n_gmm_clusters):
        for j in range(len(colour_map)):
            img[i * 100:(i + 1) * 100, j * 150:(j + 1) * 150] = colour_models[j][i]

    # Convert to BGR
    if use_lab_colour_space:
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Show image
    cv2.imshow("Colour model", img)
    cv2.waitKey(0)
