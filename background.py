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

        for frame in frames_background:
            mask_background = background_substraction.apply(frame, learningRate=0.01)
            cv2.imshow("", mask_background)
            # Pause for 1 ms
            cv2.waitKey(1)

        for frame in frames_foreground:
            mask_background = background_substraction.apply(frame, learningRate=0)
            cv2.imshow("", mask_background)
            # Pause for 1 ms
            cv2.waitKey(1)

        cv2.waitKey(0)
