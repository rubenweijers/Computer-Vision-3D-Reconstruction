import pickle
import random

import cv2
import glm
import numpy as np

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
    with open("./data/voxels.pickle", "rb") as f:
        data_pickle = pickle.load(f)

    data = []
    colours = []
    intersect = False
    if intersect:
        for frame in data_pickle["voxels"][:1]:  # TODO: Change to all frames
            intersection = set.intersection(*map(set, frame))  # Find the intersection of all sets

            for voxel in intersection:
                # voxel = [v / data_pickle["voxel_size"] for v in voxel]
                colours.append([voxel[0] / width, voxel[2] / depth, voxel[1] / height])

            data.extend(intersection)
            print(len(data))
    else:
        for frame in data_pickle["voxels"][:1]:
            for camera in frame:
                for voxel in camera:
                    # voxel = [v / data_pickle["voxel_size"] for v in voxel]  # Scale the voxel to the block size
                    voxel = [v // 30 for v in voxel]  # Scale the voxel to the block size
                    data.append(voxel)
                    colours.append([voxel[0] / width, voxel[2] / depth, voxel[1] / height])

    return data, colours


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.

    # Load camera params from pickle
    # fps_config = ["./data/cam1/config.pickle", "./data/cam2/config.pickle",
    #               "./data/cam3/config.pickle", "./data/cam4/config.pickle"]

    # cam_positions = []
    # for config in fps_config:
    #     data = load_pickle(config)
    #     intrinsics = data["intrinsics"]
    #     camera_matrix = intrinsics["camera_matrix"]
    #     distortion_coefficients = intrinsics["distortion_coefficients"]
    #     # rotation_vector = intrinsics["rotation_vectors"]
    #     # translation_vector = intrinsics["translation_vectors"]

    #     extrinsics = data["extrinsics"]
    #     rotation_vector = extrinsics["rotation_vector"]
    #     translation_vector = extrinsics["translation_vector"]

    #     transformation_matrix = np.zeros((4, 4))
    #     transformation_matrix[:3, :3] = cv2.Rodrigues(rotation_vector)[0]  # Convert vector to matrix
    #     transformation_matrix[:3, 3] = translation_vector.flatten()
    #     transformation_matrix[3, 3] = 1

    #     # C = -R^T * T
    #     # https://math.stackexchange.com/a/83578
    #     cam_position = -np.matmul(transformation_matrix[:3, :3].T, transformation_matrix[:3, 3])
    #     cam_position = cam_position.round(2)
    #     cam_positions.append(cam_position.flatten().tolist())

    cam_colours = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]
    cam_positions = [[-64 * block_size, 64 * block_size, 63 * block_size],
                     [63 * block_size, 64 * block_size, 63 * block_size],
                     [63 * block_size, 64 * block_size, -64 * block_size],
                     [-64 * block_size, 64 * block_size, -64 * block_size]]

    # print(cam_positions)
    return cam_positions, cam_colours


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]

    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])

    return cam_rotations
