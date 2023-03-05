import cv2
import glm
import numpy as np
from tqdm import trange

from data_processing import load_pickle

block_size = 1.0


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.

    # Load the voxel data from pickle
    data_pickle = load_pickle("./data/voxels.pickle")
    cam_colours = [[1.0, 0], [0, 1.0], [0, 0], [1.0, 1.0]]  # Only first two colours, last is set by height
    show_cam_colours = False

    data = []
    colours = []
    intersect = True  # Set to True to only keep voxels that are in all cameras
    pixel_values = data_pickle["pixel_values"]
    if intersect:
        for frame in data_pickle["voxels"][:1]:  # TODO: Change to all frames
            for camera_number, camera in enumerate(frame):
                camera_voxels = []
                camera_colours = []
                for voxel_number, voxel in enumerate(camera):
                    voxel = [voxel[0], -voxel[2], voxel[1]]  # Swap the y and z axis, TODO: rotate y axis by 90 degrees
                    voxel = tuple(v // data_pickle["stepsize"] * block_size for v in voxel)  # Scale the voxel by step size

                    pixel_value = pixel_values[camera_number][voxel_number]
                    pixel_value = pixel_value / 255  # Scale the pixel value to 0-1
                    # BGR to RGB
                    pixel_value = [pixel_value[2], pixel_value[1], pixel_value[0]]

                    camera_voxels.append(voxel)
                    camera_colours.append(pixel_value)

                data.append(camera_voxels)
                colours.append(camera_colours)

        # Only keep intersection of voxels that are in all four cameras
        data_filtered = []
        colours_filtered = []
        for i in trange(len(data[0])):
            if data[0][i] in data[1] and data[0][i] in data[2] and data[0][i] in data[3]:
                data_filtered.append(data[0][i])
                colours_filtered.append(colours[0][i])

        return data_filtered, colours_filtered

    else:
        for frame in data_pickle["voxels"][:1]:  # TODO: Change to all frames
            for c, camera in enumerate(frame):
                for voxel in camera:
                    # voxel = [voxel[0], voxel[2], voxel[1]]  # Swap the y and z axis, TODO: rotate y axis by 90 degrees
                    voxel = tuple(v // data_pickle["stepsize"] * block_size for v in voxel)  # Scale the voxel by step size

                    data.append(voxel)

                    if show_cam_colours:  # For debugging, set colours to camera colours
                        colours.append(cam_colours[c] + [voxel[1] / height])  # First 2 colours are set by camera, last is set by height
                    else:
                        colours.append([voxel[0] / width, voxel[2] / depth, voxel[1] / height])

    # Only keep unique voxels
    print(f"Number of voxels: {len(data)}")
    idx = np.unique(data, axis=0, return_index=True)[1]  # Get the unique indices
    data = np.array(data)[idx].tolist()
    colours = np.array(colours)[idx].tolist()
    print(f"Number of unique voxels: {len(data)}")
    return data, colours


def get_cam_positions():
    """Generates camera positions and their colours"""
    # Load camera params from pickle
    fps_config = ["./data/cam1/config.pickle", "./data/cam2/config.pickle",
                  "./data/cam3/config.pickle", "./data/cam4/config.pickle"]

    cam_positions = []
    for config in fps_config:
        data = load_pickle(config)

        extrinsics = data["extrinsics"]
        rotation_vector = extrinsics["rotation_vector"]
        translation_vector = extrinsics["translation_vector"]

        transformation_matrix = np.zeros((4, 4))
        transformation_matrix[:3, :3] = cv2.Rodrigues(rotation_vector)[0]  # Convert vector to matrix
        transformation_matrix[:3, 3] = translation_vector.flatten() / 115  # Scale the translation vector by the voxel size
        transformation_matrix[3, 3] = 1

        # C = -R^T * T https://math.stackexchange.com/a/83578
        cam_position = -np.matmul(transformation_matrix[:3, :3].T, transformation_matrix[:3, 3]) * block_size
        cam_position = cam_position.round(2).flatten().tolist()
        cam_position = [cam_position[0], -cam_position[2], cam_position[1]]  # Swap the y and z axis, rotate y axis by 90 degrees
        cam_positions.append(cam_position)

    cam_colours = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]

    print(cam_positions)
    return cam_positions, cam_colours


def get_cam_rotation_matrices():
    fps_config = ["./data/cam1/config.pickle", "./data/cam2/config.pickle",
                  "./data/cam3/config.pickle", "./data/cam4/config.pickle"]

    swap_vector = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, -1, 0, 0],
                            [0, 0, 0, 1]])

    cam_angles = []
    for config in fps_config:
        data = load_pickle(config)
        extrinsics = data["extrinsics"]
        rotation_vector = extrinsics["rotation_vector"]
        # translation_vector = extrinsics["translation_vector"]

        transformation_matrix = np.zeros((4, 4))
        transformation_matrix[:3, :3] = cv2.Rodrigues(rotation_vector)[0]  # Convert vector to matrix

        # Only first 3 rows and columns are used
        # transformation_matrix[:3, 3] = translation_vector.flatten() / 115  # Scale the translation vector by the voxel size
        # transformation_matrix[3, 3] = 1

        # Swap the y and z axis, rotate y axis by 90 degrees
        transformation_matrix = np.matmul(transformation_matrix, swap_vector)
        cam_angles.append(glm.mat4(*transformation_matrix.flatten().tolist()))

    return cam_angles
