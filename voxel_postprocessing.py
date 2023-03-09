import pickle

import numpy as np
from tqdm import trange

from assignment import block_size
from data_processing import load_pickle


def intersect_voxels():
    # Load the voxel data from pickle
    data_pickle = load_pickle("./data/voxels.pickle")
    pixel_values = data_pickle["pixel_values"]

    data = []
    colours = []
    for frame in data_pickle["voxels"][:1]:  # TODO: Change to all frames
        for camera_number, voxels in enumerate(frame):
            voxels = np.array(voxels)
            voxels = voxels[:, [0, 2, 1]]  # Swap the y and z axis
            voxels[:, 1] = -voxels[:, 1]  # Rotate y axis by 90 degrees
            voxels = voxels * block_size / data_pickle["bounds"]["stepsize"]  # Scale the voxel by step size
            voxels = voxels.round(0).astype(int)  # Round to nearest integer instead of floor
            voxels = list(map(tuple, voxels))  # Convert to list of tuples, in an efficient way

            voxel_colours = np.array(pixel_values[camera_number]) / 255  # Scale the pixel value to 0-1
            voxel_colours = voxel_colours[:, [2, 1, 0]]  # BGR to RGB
            voxel_colours = list(map(tuple, voxel_colours))  # Convert to list of tuples, in an efficient way

            data.append(voxels)
            colours.append(voxel_colours)

    # Only keep intersection of voxels that are in all four cameras
    # data_filtered = []
    # colours_filtered = []
    # for i in trange(len(data[0])):
    #     if data[0][i] in data[1] and data[0][i] in data[2] and data[0][i] in data[3]:
    #         data_filtered.append(data[0][i])
    #         colours_filtered.append(colours[0][i])

    # Get the intersection of all voxels
    voxels_filtered = data[0]
    for i in trange(1, len(data)):
        voxels_filtered = list(set(voxels_filtered).intersection(data[i]))

    # Random colour
    colours_filtered = np.random.rand(len(voxels_filtered), 3).tolist()

    return voxels_filtered, colours_filtered


if __name__ == "__main__":
    voxels, colours = intersect_voxels()

    # Save the intersection to a pickle
    with open("./data/voxels_intersection.pickle", "wb") as f:
        pickle.dump({"voxels": voxels, "colours": colours}, f)
