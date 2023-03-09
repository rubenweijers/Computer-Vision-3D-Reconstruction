import pickle
from itertools import product

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
    image_points = image_points.reshape(-1, 2)  # Flatten array

    voxel_coords = [tuple(voxel_coord) for voxel_coord in voxel_coords]  # To list of tuples
    image_points = [tuple(image_point) for image_point in image_points]

    # Convert to dict
    voxel_lookup_table = {voxel: image_point for voxel, image_point in zip(voxel_coords, image_points)}

    return voxel_lookup_table


def select_voxels(mask, voxel_lookup_table: dict, debug: bool = False) -> list:
    """Filters voxels that are visible in the image and are not masked out."""
    # if debug:  # Get max and min of image points to find outliers
    min_x = round(min([value[0] for value in voxel_lookup_table.values()]), 2)
    max_x = round(max([value[0] for value in voxel_lookup_table.values()]), 2)
    min_y = round(min([value[1] for value in voxel_lookup_table.values()]), 2)
    max_y = round(max([value[1] for value in voxel_lookup_table.values()]), 2)
    print(f"({min_x=}, {max_x=}, {min_y=}, {max_y=})")

    skipped = 0
    voxel_points = []
    image_points_all = []
    for key, value in tqdm(voxel_lookup_table.items()):
        image_points = value
        x, y, z = key

        try:
            if mask[int(image_points[1]), int(image_points[0])] == 255:  # If the image point is masked, add voxel to list
                image_points_all.append(image_points)
                voxel_points.append(key)
        except Exception as e:
            # if image_points[0] < 0 or image_points[0] >= mask.shape[0] or image_points[1] < 0 or image_points[1] >= mask.shape[1]

            skipped += 1
            if debug:
                print(f"Skipped voxel ({x=}, {y=}, {z=}) with image point ({image_points[0]:.0f}, {image_points[1]:.0f})")

    if skipped > 0:
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

    output_masks = []
    output_colours = []
    lookup_tables = []
    for camera, fp_video in enumerate(fps_background):
        frames_background = read_frames(fp_video, stop_after=2)
        if frames_background is None:
            print(f"Could not read frames from {fp_video}")
            continue

        frames_foreground = read_frames(fps_foreground[camera], stop_after=2)
        if frames_foreground is None:
            print(f"Could not read frames from {fps_foreground[camera]}")
            continue

        output_colour, output_mask = background_substraction(frames_background, frames_foreground)

        camera_params = load_pickle(fps_config[camera])

        voxel_lookup_table = make_voxel_lookup_table(camera_params, bounds)

        lookup_tables.append(voxel_lookup_table)
        output_masks.append(output_mask)
        output_colours.append(frames_foreground)

    voxels = []
    for frame_n in range(len(output_masks[0]))[:1]:  # TODO: only first two frames for debugging
        voxel_points = []
        image_points_all = []
        pixel_values_all = []
        for camera, mask in enumerate(output_masks):
            cam_voxels, image_points = select_voxels(mask[frame_n], lookup_tables[camera], debug=False)
            voxel_points.append(cam_voxels)
            # Get pixel value of all image points
            pixel_values = [output_colours[camera][frame_n][int(point[1]), int(point[0])] for point in image_points]
            pixel_values_all.append(pixel_values)
            image_points_all.append(image_points)

        print(f"Number of voxels for each camera: {[len(p) for p in voxel_points]}")

        voxels.append(voxel_points)

    # Write voxels to pickle
    with open("./data/voxels.pickle", "wb") as fp:
        pickle.dump({"voxels": voxels, "bounds": bounds, "image_points": image_points_all, "pixel_values": pixel_values_all}, fp)

    # Plot all voxels for each camera, add colour to each camera
    # plot_voxels(lookup_tables, output_colours, frame_number=0)
