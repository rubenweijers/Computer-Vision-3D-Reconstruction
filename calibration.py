import json
import pickle

import cv2
import numpy as np
from tqdm import tqdm


def make_object_points(horizontal_corners: int, vertical_corners: int, square_size: int) -> np.ndarray:
    """Make the object points for the chessboard.

    The object points are the 3D points in real world space.
    """
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(horizontal_corners-1,vertical_corners-1,0)
    objp = np.zeros((vertical_corners * horizontal_corners, 3), np.float32)
    objp[:, :2] = np.mgrid[0:vertical_corners, 0:horizontal_corners].T.reshape(-1, 2)
    objp *= square_size  # Multiply by square size to get the real world coordinates, in milimeters
    return objp


def calibrate_camera(frames: list, horizontal_corners: int, vertical_corners: int, square_size: int, fp_output: str = None, every_n: int = 50) -> dict:
    """Calibrate the camera using OpenCV.

    Uses a list of frames to calibrate the camera. The frames should be taken with the same camera and the same checkerboard.

    Based on https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    """
    # Array to store image points from all the images
    imgpoints = []  # 2D points in image plane

    # Go through training images and grayscale
    not_found = 0
    desc = f"Finding corners in every {every_n}th frame of {len(frames)} frames (total {len(frames) // every_n} frames ({len(frames) // every_n / len(frames) * 100:.2f}%))"
    for c, frame in enumerate(tqdm(frames[::every_n], desc=desc)):
        img_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        pattern_found, corners = cv2.findChessboardCorners(img_grey, (vertical_corners, horizontal_corners), None, cv2.CALIB_CB_FAST_CHECK)

        # If found, add image points (after refining them)
        if pattern_found:
            corners = cv2.cornerSubPix(img_grey, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
        else:
            not_found += 1
            # Skip 10 frames to avoid the same frame being used again

    print(f"Could not find chessboard corners in {not_found} ({not_found / (c+1) * 100:.2f}%) frames")

    # Since all images are taken with the same camera, the object points are the same
    objp = make_object_points(horizontal_corners, vertical_corners, square_size)  # 3D points in real world space
    objpoints = len(imgpoints) * [objp]  # 3D point in real world space

    # Return value, camera matrix, distortion coefficients, rotation and translation vectors
    img_grey_reshaped = img_grey.shape[::-1]
    return_val, camera_mat, dist_coef, rot_vec, transl_vec = cv2.calibrateCamera(objpoints, imgpoints, img_grey_reshaped, None, None)

    camera_mat_opt, roi = cv2.getOptimalNewCameraMatrix(camera_mat, dist_coef, img_grey_reshaped, 1, img_grey_reshaped)

    # Also save the object points and image points to calculate the calibration statistics
    camera_params = {"camera_matrix": camera_mat, "distortion_coefficients": dist_coef, "camera_matrix_optimal": camera_mat_opt, "roi": roi,
                     "rotation_vectors": rot_vec, "translation_vectors": transl_vec, "object_points": objpoints, "image_points": imgpoints}

    # Save the camera parameters to pickle file
    if fp_output is not None:
        with open(fp_output, "wb") as f:
            pickle.dump(camera_params, f)

    return camera_params


def read_frames(fp_video: str):
    """Read all frames from a video file. Returns a list of frames."""
    cap = cv2.VideoCapture(fp_video)

    if not cap.isOpened():
        print(f"Could not open {fp_video}")
        return None

    frames = []
    while True:
        available, frame = cap.read()

        if not available:
            print("Video ended")
            break

        frames.append(frame)

    cap.release()
    return frames


def read_checkerboard_xml(fp_xml: str) -> tuple:
    """Read the checkerboard dimensions from an XML file."""
    return 8, 6, 115  # TODO: Read from XML


def calculate_calibration_stats(camera_params: dict, norm: int = cv2.NORM_L2) -> dict:
    """Calculate the calibration statistics.

    Based on https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    """
    objectpoints = camera_params["object_points"]
    imagepoints = camera_params["image_points"]

    camera_mat = camera_params["camera_matrix"]
    dist_coef = camera_params["distortion_coefficients"]
    rot_vec = camera_params["rotation_vectors"]
    transl_vec = camera_params["translation_vectors"]

    errors = []
    for image in range(len(objectpoints)):
        imagepoints_proj, _ = cv2.projectPoints(objectpoints[image], rot_vec[image], transl_vec[image], camera_mat, dist_coef)
        error = cv2.norm(imagepoints[image], imagepoints_proj, norm) / len(imagepoints_proj)
        errors.append(error)

    return {"total_images": len(objectpoints),
            "mean_error": sum(errors) / len(objectpoints),
            "individual_errors": errors}


if __name__ == "__main__":
    # Calibration mode
    fps_videos = ["./data/cam1/intrinsics.avi", "./data/cam2/intrinsics.avi", "./data/cam3/intrinsics.avi", "./data/cam4/intrinsics.avi"]
    fps_camera_params = ["./data/cam1/camera_params.pickle", "./data/cam2/camera_params.pickle",
                         "./data/cam3/camera_params.pickle", "./data/cam4/camera_params.pickle"]
    fps_stats = ["./data/cam1/stats.json", "./data/cam2/stats.json", "./data/cam3/stats.json", "./data/cam4/stats.json"]
    fp_xml = "./data/checkerboard.xml"

    # Read XML with checkerboard parameters
    horizontal_corners, vertical_corners, square_size = read_checkerboard_xml(fp_xml)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for c, fp_video in enumerate(fps_videos):
        frames = read_frames(fp_video)
        if frames is None:
            print(f"Could not read frames from {fp_video}")
            continue

        camera_params = calibrate_camera(frames, horizontal_corners, vertical_corners, square_size, fps_camera_params[c])

        stats = calculate_calibration_stats(camera_params)
        print(f"The mean error of camera {c+1} is {stats['mean_error']:.2f}")

        # Save the stats
        with open(fps_stats[c], "w") as f:
            json.dump(stats, f, indent=4)
