import pickle

import cv2
import numpy as np
from tqdm import trange

from background import background_substraction
from calibration import read_frames
from data_processing import load_pickle


def make_voxel_lookup_table(camera_params, lowerbound=-750, upperbound=750, stepsize=30):
    # Intrinsics
    intrinsics = camera_params["intrinsics"]
    camera_matrix = np.array(intrinsics["camera_matrix"], dtype=np.float32)
    distortion_coefficients = np.array(intrinsics["distortion_coefficients"], dtype=np.float32)

    # Extrinsics
    extrinsics = camera_params["extrinsics"]
    rotation_vector = np.array(extrinsics["rotation_vector"], dtype=np.float32)
    translation_vector = np.array(extrinsics["translation_vector"], dtype=np.float32)

    voxel_lookup_table = {}
    for x in trange(lowerbound, upperbound, stepsize, desc="Generating voxel lookup table"):
        for y in range(0, upperbound*2, stepsize):
            for z in range(lowerbound, upperbound, stepsize):
                voxel = np.array([x, y, z], dtype=np.float32)  # Real-world coordinates

                image_points, jac = cv2.projectPoints(voxel, rotation_vector, translation_vector,
                                                      camera_matrix, distortion_coefficients)  # 2D image coordinates

                voxel_lookup_table[(x, y, z)] = image_points.flatten()

    return voxel_lookup_table


def select_voxels(frame, voxel_lookup_table, lowerbound=-750, upperbound=750, stepsize=30, debug=False):
    voxel_points = []
    skipped = 0

    for x in trange(lowerbound, upperbound, stepsize, desc="Selecting voxels"):
        for y in range(0, upperbound*2, stepsize):
            for z in range(lowerbound, upperbound, stepsize):
                image_points = voxel_lookup_table[(x, y, z)]

                # Skip if out of bounds
                if image_points[0] >= frame.shape[0] or image_points[1] >= frame.shape[1] or image_points[0] < 0 or image_points[1] < 0:
                    skipped += 1
                    if debug:
                        print(f"Skipped voxel ({x}, {y}, {z}) with image point ({image_points[0]:.0f}, {image_points[1]:.0f})")
                    continue

                if frame[int(image_points[0]), int(image_points[1])] == 255:  # If the image point is masked, add voxel to list
                    voxel_points.append((x, y, z))

    if skipped > 0:
        print(f"Skipped {skipped} voxels ({skipped / (len(voxel_lookup_table) / 100):.2f}%)")
    return voxel_points


if __name__ == "__main__":
    # Calibration mode
    fps_background = ["./data/cam1/background.avi", "./data/cam2/background.avi",
                      "./data/cam3/background.avi", "./data/cam4/background.avi"]
    fps_foreground = ["./data/cam1/video.avi", "./data/cam2/video.avi",
                      "./data/cam3/video.avi", "./data/cam4/video.avi"]
    fps_config = ["./data/cam1/config.pickle", "./data/cam2/config.pickle",
                  "./data/cam3/config.pickle", "./data/cam4/config.pickle"]
    fp_xml = "./data/checkerboard.xml"

    lowerbound = -1500
    upperbound = 1500
    stepsize = 30  # mm
    voxel_size = 115  # mm

    output_masks = []
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
        voxel_lookup_table = make_voxel_lookup_table(camera_params, lowerbound=lowerbound, upperbound=upperbound, stepsize=stepsize)

        lookup_tables.append(voxel_lookup_table)
        output_masks.append(output_mask)

    voxels = []
    for frame_n in range(len(output_masks[0]))[:1]:  # TODO: only first two frames for debugging
        voxel_points = []
        for camera, mask in enumerate(output_masks):
            voxel_points.append(select_voxels(mask[frame_n], lookup_tables[camera],
                                              lowerbound=lowerbound, upperbound=upperbound, stepsize=stepsize, debug=False))

        print(f"Number of voxels for each camera: {[len(p) for p in voxel_points]}")

        voxels.append(voxel_points)

    # Write voxels to pickle
    with open("./data/voxels.pickle", "wb") as fp:
        pickle.dump({"voxels": voxels, "voxel_size": voxel_size,
                     "lowerbound": lowerbound, "upperbound": upperbound, "stepsize": stepsize}, fp)
