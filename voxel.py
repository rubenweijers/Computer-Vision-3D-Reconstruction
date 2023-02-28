import cv2
import numpy as np

from background import background_substraction
from calibration import make_object_points, read_frames
from data_processing import load_pickle


def make_voxel_lookup_table(camera_params, voxel_size=115, number_of_voxels=1150):
    # Intrinsics
    intrinsics = camera_params["intrinsics"]
    camera_matrix = np.array(intrinsics["camera_matrix"], dtype=np.float32)
    distortion_coefficients = np.array(intrinsics["distortion_coefficients"], dtype=np.float32)

    # Extrinsics
    extrinsics = camera_params["extrinsics"]
    rotation_vectors = np.array(extrinsics["rotation_vector"], dtype=np.float32)
    translation_vectors = np.array(extrinsics["translation_vector"], dtype=np.float32)

    voxel_lookup_table = {}
    for x in range(0, number_of_voxels, voxel_size):
        for y in range(0, number_of_voxels, voxel_size):
            for z in range(0, number_of_voxels, voxel_size):
                voxel = np.array([x, y, z], dtype=np.float32)  # Real-world coordinates

                image_points, jac = cv2.projectPoints(voxel, rotation_vectors, translation_vectors,
                                                      camera_matrix, distortion_coefficients)  # 2D image coordinates

                voxel_lookup_table[(x, y, z)] = image_points.flatten()

    return voxel_lookup_table


def select_voxels(frame, voxel_lookup_table, voxel_size=115, number_of_voxels=1150):
    voxel_points = []
    skipped = 0

    for x in range(0, number_of_voxels, voxel_size):
        for y in range(0, number_of_voxels, voxel_size):
            for z in range(0, number_of_voxels, voxel_size):
                image_points = voxel_lookup_table[(x, y, z)]

                # Skip if out of bounds
                if image_points[0] >= frame.shape[0] or image_points[1] >= frame.shape[1]:
                    skipped += 1
                    continue

                if frame[int(image_points[0]), int(image_points[1])] == 255:  # If the image point is masked, add voxel to list
                    voxel_points.append((x, y, z))

    print(f"Skipped {skipped} voxels")
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

    output_masks = []
    lookup_tables = []
    for camera, fp_video in enumerate(fps_background):
        frames_background = read_frames(fp_video)
        if frames_background is None:
            print(f"Could not read frames from {fp_video}")
            continue

        frames_foreground = read_frames(fps_foreground[camera])
        if frames_foreground is None:
            print(f"Could not read frames from {fps_foreground[camera]}")
            continue

        output_colour, output_mask = background_substraction(frames_background, frames_foreground)

        camera_params = load_pickle(fps_config[camera])
        voxel_lookup_table = make_voxel_lookup_table(camera_params)

        lookup_tables.append(voxel_lookup_table)
        output_masks.append(output_mask)

    for frame_n in range(len(output_masks[0])):
        voxel_points = []
        for camera, mask in enumerate(output_masks):
            voxel_points += select_voxels(mask[frame_n], lookup_tables[camera])

        print(len(voxel_points), len(set(voxel_points)))
