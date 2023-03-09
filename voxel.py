import pickle
from itertools import product

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from tqdm.contrib import tzip

from background import background_substraction
from calibration import read_frames
from data_processing import load_pickle


def make_voxel_lookup_table(camera_params: dict, bounds: dict) -> dict:
    # Intrinsics
    intrinsics = camera_params["intrinsics"]
    camera_matrix = intrinsics["camera_matrix"]
    distortion_coefficients = intrinsics["distortion_coefficients"]

    # Extrinsics
    extrinsics = camera_params["extrinsics"]
    rotation_vector = extrinsics["rotation_vector"]
    translation_vector = extrinsics["translation_vector"]

    voxel_coords = product(range(bounds["x_lowerbound"], bounds["x_upperbound"], bounds["stepsize"]),
                           range(bounds["y_lowerbound"], bounds["y_upperbound"], bounds["stepsize"]),
                           range(bounds["z_lowerbound"], bounds["z_upperbound"], bounds["stepsize"]))
    voxel_coords = np.array(list(voxel_coords), dtype=np.float32)  # 3D real world coordinates

    image_points, jac = cv2.projectPoints(voxel_coords, rotation_vector, translation_vector,
                                          camera_matrix, distortion_coefficients)  # 2D image coordinates
    image_points = image_points.reshape(-1, 2).astype(np.int32)

    # Threshold all points that are out of bounds TODO: add upper bounds to remove try-except in select_voxels
    idx = np.where((image_points[:, 0] < 0) | (image_points[:, 1] < 0))
    image_points = np.delete(image_points, idx, axis=0)
    voxel_coords = np.delete(voxel_coords, idx, axis=0)

    voxel_coords = [tuple(voxel_coord) for voxel_coord in voxel_coords]  # To list of tuples
    image_points = [tuple(image_point) for image_point in image_points]

    # Convert to dict
    voxel_lookup_table = {voxel: image_point for voxel, image_point in zip(voxel_coords, image_points)}

    return voxel_lookup_table


def select_voxels(mask, voxel_lookup_table: dict, debug: bool = False) -> list:
    """Filters voxels that are visible in the image and are not masked out."""
    skipped = 0
    voxel_points = []
    image_points_all = []
    for key, value in voxel_lookup_table.items():
        image_points = value

        try:
            if mask[image_points[1], image_points[0]] == 255:  # If the image point is masked, add voxel to list
                image_points_all.append(image_points)
                voxel_points.append(key)
        except Exception as e:
            skipped += 1
            continue

    if debug:
        print(f"{skipped} voxels out of bounds ({skipped / (len(voxel_lookup_table) / 100):.2f}%)")

    return voxel_points, image_points_all


def plot_voxels(lookup_tables: list, frame: list, frame_number: int = 0):
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

    for i, lookup_table in enumerate(lookup_tables):
        values = np.array(list(lookup_table.values()))

        # Clip outliers
        values = values[values[:, 0] >= 0]
        values = values[values[:, 1] >= 0]
        values = values[values[:, 0] < frame[i][frame_number].shape[1]]
        values = values[values[:, 1] < frame[i][frame_number].shape[0]]

        # Draw points on first frame
        # TODO: Move axis from opencv to matplotlib coordinates
        for point in values:
            # PROBLEM: voxels are not drawn where they are supposed to be
            cv2.circle(frame[i][frame_number], (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)

        # cv2.imshow(f"Camera {i+1}", frame[i][frame_number])
        # cv2.waitKey(0)

        frame[i][frame_number] = cv2.cvtColor(frame[i][frame_number], cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
        axs[i // 2, i % 2].imshow(frame[i][frame_number])

        axs[i // 2, i % 2].scatter(values[:, 0], values[:, 1], s=1, c="r")
        axs[i // 2, i % 2].set_title(f"Camera {i+1}")

    plt.show()


if __name__ == "__main__":
    fps_background = ["./data/cam1/background.avi", "./data/cam2/background.avi",
                      "./data/cam3/background.avi", "./data/cam4/background.avi"]
    fps_foreground = ["./data/cam1/video.avi", "./data/cam2/video.avi",
                      "./data/cam3/video.avi", "./data/cam4/video.avi"]
    fps_config = ["./data/cam1/config.pickle", "./data/cam2/config.pickle",
                  "./data/cam3/config.pickle", "./data/cam4/config.pickle"]
    fp_xml = "./data/checkerboard.xml"

    bounds = {"x_lowerbound": -1000, "x_upperbound": 4000,
              "y_lowerbound": -1000, "y_upperbound": 3000,
              "z_lowerbound": -2200, "z_upperbound": 0,
              "stepsize": 30, "voxel_size": 115}  # mm

    n_frames = 5  # Number of frames in total to load
    every_nth = 50  # Only load every nth frame

    all_masks = []
    all_frames = []
    all_tables = []
    for fp_back, fp_fore, fp_config in tzip(fps_background, fps_foreground, fps_config, desc="Loading camera data", unit="camera"):
        frames_background = read_frames(fp_back, stop_after=n_frames, nth=every_nth)
        if frames_background is None:
            print(f"Could not read frames from {fp_back}")
            continue

        frames_foreground = read_frames(fp_fore, stop_after=n_frames, nth=every_nth)
        if frames_foreground is None:
            print(f"Could not read frames from {fp_fore}")
            continue

        masks_coloured, masks = background_substraction(frames_background, frames_foreground)

        camera_params = load_pickle(fp_config)
        voxel_lookup_table = make_voxel_lookup_table(camera_params, bounds)

        all_frames.append(frames_foreground)
        all_masks.append(masks)
        all_tables.append(voxel_lookup_table)

    all_voxels = []  # List of voxels for each frame
    all_pixel_values = []
    for frame_n in trange(n_frames, desc="Processing frames", unit="frame"):
        voxels_frame = []
        pixel_values_frame = []

        for camera_masks, camera_table, camera_frames in zip(all_masks, all_tables, all_frames):
            voxels_camera, image_points_camera = select_voxels(camera_masks[frame_n], camera_table)

            # Rewrite to numpy array
            pixel_values = np.zeros((len(image_points_camera), 3))
            cf = camera_frames[frame_n]
            points = np.array(image_points_camera)
            points = points[:, 1], points[:, 0]
            pixel_values = cf[points]

            # pixel_values = [camera_frames[frame_n][point[1], point[0]] for point in image_points_camera]

            voxels_frame.append(voxels_camera)
            pixel_values_frame.append(pixel_values)

        all_voxels.append(voxels_frame)
        all_pixel_values.append(pixel_values_frame)

    # Write voxels to pickle
    with open("./data/voxels.pickle", "wb") as fp:
        pickle.dump({"voxels": all_voxels, "pixel_values": all_pixel_values, "bounds": bounds}, fp)

    # Plot all voxels for each camera, add colour to each camera
    # plot_voxels(lookup_tables, output_colours, frame_number=0)
