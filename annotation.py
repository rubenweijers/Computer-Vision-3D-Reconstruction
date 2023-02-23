import os
import pickle

import cv2
import numpy as np

from calibration import read_checkerboard_xml, read_frames


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

    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerSubPix(img_grey, corners, (11, 11), (-1, -1), criteria)

    return corners


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
        cv2.circle(frame, (x, y), 3, (0, 165, 255), -1)
        cv2.imshow("", frame)

        points.append((x_corrected, y_corrected))
        if len(points) < 4:
            print(points_d[len(points)])

    # After the 4th cornerpoint ...
    if (len(points) == 4) and (event == cv2.EVENT_LBUTTONDOWN):
        print("All corners have been provided!")

        corners = interpolate_points(points, frame)

        # Check if pickle file exists, to save annotations
        if not os.path.exists(param["fp_output"]):
            all_points = {}
        else:
            with open(param["fp_output"], "rb") as f:
                all_points = pickle.load(f)

        # Add new points, overwriting old annotations if they exist
        all_points[param["frame_number"]] = corners

        # Save pickle dict
        with open(param["fp_output"], "wb") as f:
            pickle.dump(all_points, f)

        # Draw the interpolated points
        for corner in corners:
            cv2.circle(frame, (corner[0] * zoom).astype(int), 2, (0, 0, 255), -1)

        cv2.imshow("", frame)


if __name__ == "__main__":
    # Loading data
    fps_videos = ["./data/cam1/checkerboard.avi", "./data/cam2/checkerboard.avi",
                  "./data/cam3/checkerboard.avi", "./data/cam4/checkerboard.avi"]
    fps_camera_params = ["./data/cam1/camera_params.pickle", "./data/cam2/camera_params.pickle",
                         "./data/cam3/camera_params.pickle", "./data/cam4/camera_params.pickle"]
    fps_annotations = ["./data/cam1/annotations.pickle", "./data/cam2/annotations.pickle",
                       "./data/cam3/annotations.pickle", "./data/cam4/annotations.pickle"]
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
        cv2.setMouseCallback("", on_click, param={"fp_output": fps_annotations[c], "frame_number": frame_number, "zoom": zoom})

        cv2.waitKey(0)
        cv2.destroyAllWindows()
