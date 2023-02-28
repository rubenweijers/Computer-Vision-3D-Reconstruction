import cv2
import numpy as np

from calibration import read_frames


def background_substraction(frames_background, frames_foreground) -> list:
    """Background substraction using OpenCV's MOG2 algorithm."""
    # Calculate the average frame
    background_substraction = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=True)

    # Training phase
    for frame in frames_background:
        _ = background_substraction.apply(frame, learningRate=0.01)

    # Inference phase
    output_colour = []
    output_mask = []
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

        output_colour.append(frame)
        output_mask.append(mask_foreground)

    return output_colour, output_mask


if __name__ == "__main__":
    fps_backgrounds = ["./data/cam1/background.avi", "./data/cam2/background.avi",
                       "./data/cam3/background.avi", "./data/cam4/background.avi"]
    fps_foreground = ["./data/cam1/video.avi", "./data/cam2/video.avi",
                      "./data/cam3/video.avi", "./data/cam4/video.avi"]

    for c, fp_video in enumerate(fps_backgrounds):
        frames_background = read_frames(fp_video)
        if frames_background is None:
            print(f"Could not read frames from {fp_video}")
            continue

        frames_foreground = read_frames(fps_foreground[c])
        if frames_foreground is None:
            print(f"Could not read frames from {fps_foreground[c]}")
            continue

        avg_frame = np.mean(frames_background, axis=0).astype(np.uint8)  # TODO: Change to GMM
        print(f"Average frame shape: {avg_frame.shape}")

        # Show the average frame
        cv2.imshow("", avg_frame)
        output_colour, output_mask = background_substraction(frames_background, frames_foreground)
        cv2.waitKey(0)

        for frame in output_colour:
            cv2.imshow("", frame)
            cv2.waitKey(1)

        cv2.waitKey(0)
