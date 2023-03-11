import cv2
import glm
import numpy as np

from data_processing import load_pickle

block_size = 1.0
frame_n = 0
data = load_pickle("./data/voxels_intersection.pickle")  # Load data only once
voxels = data["voxels"]
colours = data["colours"]
bounds = data["bounds"]
torso_only = True


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
    global frame_n

    frame_voxels = voxels[frame_n]
    frame_colours = colours[frame_n]
    frame_n = (frame_n + 1) % len(voxels)  # Reset to frame 0 when the last frame is reached

    if torso_only:  # Only select voxels between 85 and 150 cm
        hip_height = 850  # mm
        shoulder_height = 1500  # mm

        frame_voxels = np.array(frame_voxels) * bounds["stepsize"]  # Convert back to mm
        idx = np.where((frame_voxels[:, 1] >= hip_height) & (frame_voxels[:, 1] <= shoulder_height))  # Order: x, z, y

        frame_voxels = frame_voxels[idx] / bounds["stepsize"]  # Select voxels between 85 and 150 cm
        frame_voxels = frame_voxels.round().astype(int).tolist()  # Round to nearest integer and convert to list
        frame_colours = np.array(frame_colours)[idx].tolist()  # Index colours with the same indices as the voxels

    return frame_voxels, frame_colours


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
