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
        for camera_number, camera in enumerate(frame):
            camera_voxels = []
            camera_colours = []
            for voxel_number, voxel in enumerate(camera):  # TODO: make use of numpy
                voxel = [voxel[0], -voxel[2], voxel[1]]  # Swap the y and z axis, rotate y axis by 90 degrees
                voxel = tuple(v / data_pickle["bounds"]["stepsize"] * block_size for v in voxel)  # Scale the voxel by step size

                pixel_value = pixel_values[camera_number][voxel_number]
                pixel_value = pixel_value / 255  # Scale the pixel value to 0-1
                # BGR to RGB
                pixel_value = (pixel_value[2], pixel_value[1], pixel_value[0])

                camera_voxels.append(voxel)
                camera_colours.append(pixel_value)

            data.append(camera_voxels)
            colours.append(camera_colours)

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
