# Read all frames

import cv2
import numpy as np

from calibration import read_checkerboard_xml, read_frames
from data_processing import load_pickle

if __name__ == "__main__":
    # Calibration mode
    fps_backgrounds = ["./data/cam1/background.avi", "./data/cam2/background.avi",
                       "./data/cam3/background.avi", "./data/cam4/background.avi"]
    fps_videos = ["./data/cam1/video.avi", "./data/cam2/video.avi",
                  "./data/cam3/video.avi", "./data/cam4/video.avi"]
    fps_config = ["./data/cam1/config.pickle", "./data/cam2/config.pickle",
                  "./data/cam3/config.pickle", "./data/cam4/config.pickle"]
    fp_xml = "./data/checkerboard.xml"

    for c, fp_video in enumerate(fps_backgrounds):
        frames_background = read_frames(fp_video)
        if frames_background is None:
            print(f"Could not read frames from {fp_video}")
            continue

        frames_foreground = read_frames(fps_videos[c])
        if frames_foreground is None:
            print(f"Could not read frames from {fps_videos[c]}")
            continue

        camera_params = load_pickle(fps_config[c])

        # Calculate the average frame
        frames_background = np.array(frames_background)
        avg_frame = np.mean(frames_background, axis=0).astype(np.uint8)  # TODO: Change to GMM
        print(f"Average frame shape: {avg_frame.shape}")

        # Show the average frame
        cv2.imshow("", avg_frame)
        cv2.waitKey(0)

        background_substraction = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=True)

        # Training phase
        for frame in frames_background:
            mask_background = background_substraction.apply(frame, learningRate=0.01)
            # cv2.imshow("", mask_background)
            # Pause for 1 ms
            # cv2.waitKey(1)

        # Inference phase
        for frame in frames_foreground:
            mask_foreground = background_substraction.apply(frame, learningRate=0)
            # Remove shadow from the mask
            mask_foreground[mask_foreground == 127] = 0  # 0: background, 255: foreground, 127: shadow

            # Find biggest contour in the mask
            # TODO: make sure gaps, e.g. between the legs, are not filled
            all_contours, _ = cv2.findContours(mask_foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(all_contours) > 0:  # If there are any contours
                contour_max = max(all_contours, key=cv2.contourArea)
                # Draw the contour
                # cv2.drawContours(frame, [contour_max], -1, (0, 255, 0), 2)

            # Only keep pixels within the contour
            mask_contour = np.zeros(mask_foreground.shape, dtype=np.uint8)
            cv2.drawContours(mask_contour, [contour_max], -1, 255, -1)
            mask_foreground = cv2.bitwise_and(mask_foreground, mask_contour)

            # Apply erosion and dilation to remove noise
            kernel = np.ones((3, 3), np.uint8)
            mask_foreground = cv2.dilate(mask_foreground, kernel, iterations=1)
            mask_foreground = cv2.erode(mask_foreground, kernel, iterations=1)

            frame = cv2.bitwise_and(frame, frame, mask=mask_foreground)

            cv2.imshow("", frame)
            cv2.waitKey(1)

        cv2.waitKey(0)
