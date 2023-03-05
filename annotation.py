import os
import pickle

import cv2
import numpy as np

from calibration import make_object_points, read_checkerboard_xml, read_frames


def draw(img, corners, imagepoints):
    """Draws the axis on the image.

    Based on https://docs.opencv.org/4.6.0/d7/d53/tutorial_py_pose.html
    """
    corners = corners.astype(int)
    origin = corners[0].reshape(2)  # Flattening the array
    imagepoints = imagepoints.astype(int).reshape(-1, 2)
    colour = (0, 165, 255)  # Orange

    # Draw a cube at the origin
    for start, end in zip(range(4), range(4, 8)):
        # Draw the lines between the corners
        img = cv2.line(img, imagepoints[start], imagepoints[end], colour, 3)

    img = cv2.drawContours(img, [imagepoints[4:]], -1, colour, 3)  # Top face
    img = cv2.drawContours(img, [imagepoints[:4]], -1, colour, 3)  # Bottom face

    # Draw the three main axis
    img = cv2.line(img, origin, imagepoints[3], (255, 0, 0), 10)  # Draw x axis in blue
    img = cv2.line(img, origin, imagepoints[1], (0, 255, 0), 10)  # Draw y axis in green
    img = cv2.line(img, origin, imagepoints[4], (0, 0, 255), 10)  # Draw z axis in red

    # Draw big circle at origin
    img = cv2.circle(img, origin, 7, colour, -1)

    # Draw black circles at imagepoints
    for start in imagepoints:
        img = cv2.circle(img, start, 3, (0, 0, 0), -1)

    return img


def interpolate_points(points, img):
    """Interpolate the points to get the corners of the chessboard."""
    x1, y1 = points[0]  # left-top
    x2, y2 = points[1]  # right-top
    x3, y3 = points[2]  # left-bottom
    x4, y4 = points[3]  # right-bottom

    x1x2 = np.linspace(x1, x2, horizontal_corners)  # Interpolate the x-coordinates of the top row
    x3x4 = np.linspace(x3, x4, horizontal_corners)  # Interpolate the x-coordinates of the bottom row

    y1y3 = np.linspace(y1, y3, vertical_corners)  # Interpolate the y-coordinates of the left column
    y2y4 = np.linspace(y2, y4, vertical_corners)  # Interpolate the y-coordinates of the right column

    corners = np.zeros((vertical_corners, horizontal_corners, 2), dtype=np.float32)  # 2D array of all corners

    # TODO: Add getPerspectiveTransform()

    for v in range(vertical_corners):
        weight_vertical_inv = v / (vertical_corners - 1)  # From 0 to 1 (minus 1 because we start at 0)
        weight_vertical = 1 - weight_vertical_inv  # From 1 to 0

        for h in range(horizontal_corners):
            # Apply weighting to the x and y coordinates
            # The closer the point is to the top or left, the more weight it gets

            weight_horizontal_inv = h / (horizontal_corners - 1)  # From 0 to 1 (minus 1 because we start at 0)
            weight_horizontal = 1 - weight_horizontal_inv  # From 1 to 0

            x = weight_vertical * x1x2[h] + weight_vertical_inv * x3x4[h]
            y = weight_horizontal * y1y3[v] + weight_horizontal_inv * y2y4[v]
            corners[v, h] = (x, y)

    # Check if estimated corners are the same as the provided corners
    est_corners = np.array([corners[0, 0], corners[0, -1], corners[-1, 0], corners[-1, -1]])
    assert np.array_equal(np.array(points), est_corners), "The four corners are not the same!"

    # Reshape corners to a 2D array
    corners = corners.reshape(-1, 1, 2)

    # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # corners = cv2.cornerSubPix(img_grey, corners, (11, 11), (-1, -1), criteria)

    return corners


def calculate_extrinsics(corners: np.ndarray, camera_params: dict, square_size: int, horizontal_corners: int, vertical_corners: int) -> tuple:
    # Find the rotation and translation vectors, load camera parameters from pickle file
    print("Find the rotation and translation vectors")

    # Calculate object points
    object_points = make_object_points(horizontal_corners, vertical_corners, square_size)

    # useExtrinsicGuess = False since we don't know the initial rotation and translation vectors
    pattern_found, rot_vec, transl_vec = cv2.solvePnP(object_points, corners,
                                                      camera_params["camera_matrix"], camera_params["distortion_coefficients"],
                                                      useExtrinsicGuess=False)

    n = square_size * 3  # length of the axis in mm
    axis = np.float32([[0, 0, 0],
                       [0, n, 0],
                       [n, n, 0],
                       [n, 0, 0],
                       [0, 0, -n],
                       [0, n, -n],
                       [n, n, -n],
                       [n, 0, -n]])

    print("Project 3D points to image plane")
    imagepoints, jac = cv2.projectPoints(axis, rot_vec, transl_vec,
                                         camera_params["camera_matrix"], camera_params["distortion_coefficients"])

    return rot_vec, transl_vec, imagepoints


def on_click(event, x, y, flags, param) -> None:
    """Callback function for mouse events.

    First, collects the four corners of the chessboard provided by the user.
    Then, interpolates the points to get the corners of the chessboard.
    """
    if (len(points) < 4) and (event == cv2.EVENT_LBUTTONDOWN):  # If not all corners have been provided and the left mouse button is clicked
        x_corrected = int(x / zoom)  # Correct for zoom
        y_corrected = int(y / zoom)

        print(f"{x=}, {y=}; corrected for zoom: {x_corrected=}, {y_corrected=}")

        # Draw an orange circle on the clicked point
        cv2.circle(params["frame"], (x, y), 3, (0, 165, 255), -1)
        cv2.imshow("", params["frame"])

        points.append((x_corrected, y_corrected))
        if len(points) < 4:
            print(points_d[len(points)])

    # After the 4th cornerpoint ...
    if (len(points) == 4) and (event == cv2.EVENT_LBUTTONDOWN):
        print("All corners have been provided!")

        corners = interpolate_points(points, params["frame"])

        # Check if pickle file exists, to save annotations
        if not os.path.exists(param["fp_output"]):
            all_points = {}
        else:
            with open(param["fp_output"], "rb") as f:  # Append to existing pickle file, overwrite old annotations
                all_points = pickle.load(f)

        # Add new points, overwriting old annotations if they exist
        all_points[param["frame_number"]] = corners

        # Save pickle dict
        with open(param["fp_output"], "wb") as f:
            pickle.dump(all_points, f)

        # Draw the interpolated points
        for corner in corners:
            cv2.circle(params["frame"], (corner[0] * zoom).astype(int), 2, (0, 0, 255), -1)

        rot_vec, transl_vec, imagepoints = calculate_extrinsics(
            corners, param["camera_params"], param["square_size"], param["horizontal_corners"], param["vertical_corners"])

        # Save extrinsics to pickle file
        extrinsics = {"rotation_vector": rot_vec, "translation_vector": transl_vec, "imagepoints": imagepoints}
        with open(param["fp_extrinsics"], "wb") as f:
            pickle.dump(extrinsics, f)

        # Draw the axis on the image
        print("Draw the axis on the image")
        params["frame"] = draw(params["frame"], corners * zoom, imagepoints * zoom)
        cv2.imshow("", params["frame"])


if __name__ == "__main__":
    # Loading data
    fps_videos = ["./data/cam1/checkerboard.avi", "./data/cam2/checkerboard.avi",
                  "./data/cam3/checkerboard.avi", "./data/cam4/checkerboard.avi"]
    fps_camera_params = ["./data/cam1/camera_params.pickle", "./data/cam2/camera_params.pickle",
                         "./data/cam3/camera_params.pickle", "./data/cam4/camera_params.pickle"]
    fps_annotations = ["./data/cam1/annotations.pickle", "./data/cam2/annotations.pickle",
                       "./data/cam3/annotations.pickle", "./data/cam4/annotations.pickle"]
    fps_extrinsics = ["./data/cam1/extrinsics.pickle", "./data/cam2/extrinsics.pickle",
                      "./data/cam3/extrinsics.pickle", "./data/cam4/extrinsics.pickle"]
    fp_xml = "./data/checkerboard.xml"

    # Read XML with checkerboard parameters
    horizontal_corners, vertical_corners, square_size = read_checkerboard_xml(fp_xml)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Annotation params
    zoom = 1.5
    points_d = {0: "Provide the left-top corner of the checkerboard",
                1: "Provide the right-top corner of the checkerboard",
                2: "Provide the left-bottom corner of the checkerboard",
                3: "Provide the right-bottom corner of the checkerboard"}

    for c, fp_video in enumerate(fps_videos):
        frames = read_frames(fp_video)
        if frames is None:
            print(f"Could not read frames from {fp_video}")
            continue

        # Load camera parameters
        camera_params = pickle.load(open(fps_camera_params[c], "rb"))

        # Get the middle frame of the video
        frame_number = len(frames) // 2
        frame = frames[frame_number]

        # Annotate the frame
        points = []
        frame = cv2.resize(frame, (0, 0), fx=zoom, fy=zoom)

        cv2.imshow("", frame)
        params = {"fp_output": fps_annotations[c], "fp_extrinsics": fps_extrinsics[c], "frame_number": frame_number, "zoom": zoom,
                  "camera_params": camera_params, "horizontal_corners": horizontal_corners, "vertical_corners": vertical_corners,
                  "square_size": square_size, "frame": frame}
        cv2.setMouseCallback("", on_click, param=params)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
