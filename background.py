import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter

from calibration import read_frames

# Calculate the most common RGB value for each camera
def get_most_common_rgb_value(fp_video: str) -> tuple:
    """Calculate the most common RGB value for each camera."""
    frames = read_frames(fp_video)
    if frames is None:
        print(f"Could not read frames from {fp_video}")
        return None

    # Calculate the average frame
    avg_frame = np.mean(frames, axis=0).astype(np.uint8)  # TODO: Change to GMM
    print(f"Average frame shape: {avg_frame.shape}")

    # Show the average frame
    cv2.imshow("", avg_frame)
    cv2.waitKey(0)

    # Convert the average frame to RGB
    avg_frame = cv2.cvtColor(avg_frame, cv2.COLOR_BGR2RGB)
    # Flatten the average frame
    avg_frame = avg_frame.reshape((avg_frame.shape[0] * avg_frame.shape[1], 3))
    # Cluster the pixels
    clt = KMeans(n_clusters=1)
    clt.fit(avg_frame)
    # Count the most common pixel value
    count = Counter(clt.labels_)
    # Get the most common pixel value
    pixelvalue = clt.cluster_centers_[count.most_common(1)[0][0]].astype(int)
    return pixelvalue

# Get the most common RGB value for each camera
most_common_rgb_value_cam1 = get_most_common_rgb_value("./data/cam1/background.avi")
most_common_rgb_value_cam2 = get_most_common_rgb_value("./data/cam2/background.avi")
most_common_rgb_value_cam3 = get_most_common_rgb_value("./data/cam3/background.avi")
most_common_rgb_value_cam4 = get_most_common_rgb_value("./data/cam4/background.avi")

# Find similar RGB values to the most common RGB values of the four cameras
def find_similar_rgb_values(most_common_rgb_value: tuple, threshold: int = 5) -> list:
    """Find similar RGB values to the most common RGB value of the four cameras."""
    similar_rgb_values = []
    for i in range(256):
        for j in range(256):
            for k in range(256):
                if (abs(i - most_common_rgb_value[0]) < threshold and
                    abs(j - most_common_rgb_value[1]) < threshold and
                    abs(k - most_common_rgb_value[2]) < threshold):
                    similar_rgb_values.append([i, j, k])
    return similar_rgb_values


# Find the most similar RGB values for camera 1
similar_rgb_cam1 = find_similar_rgb_values(most_common_rgb_value_cam1)
similar_rgb_cam2 = find_similar_rgb_values(most_common_rgb_value_cam2)
similar_rgb_cam3 = find_similar_rgb_values(most_common_rgb_value_cam3)
similar_rgb_cam4 = find_similar_rgb_values(most_common_rgb_value_cam4)

# Plot the most common RGB value for each camera
def plot_most_common_rgb_value(most_common_rgb_value: tuple, similar_rgb_values: list):
    """Plot the most common RGB value for each camera."""
    # Plot the most common RGB value
    plt.figure()
    plt.imshow(np.array(most_common_rgb_value).reshape(1, 1, 3))
    plt.title("Most common RGB value")
    plt.show()

    # Plot the similar RGB values
    plt.figure()
    plt.imshow(np.array(similar_rgb_values).reshape(len(similar_rgb_values), 1, 3))
    plt.title("Similar RGB values")
    plt.show()

def background_substraction(frames_background, frames_foreground, n_contours: int = 4, min_area: int = 3000) -> list:
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
            # Filter contours with area < min_area
            contours_max = filter(lambda contour: cv2.contourArea(contour) > min_area, all_contours)
            # Return the n largest contours
            contours_max = sorted(contours_max, key=cv2.contourArea, reverse=True)[:n_contours]
            # Draw the contour
            # cv2.drawContours(frame, [contour_max], -1, (0, 255, 0), 2)

        # Only keep pixels within the contour
        mask_contour = np.zeros(mask_foreground.shape, dtype=np.uint8)
        cv2.drawContours(mask_contour, contours_max, -1, 255, -1)
        mask_foreground = cv2.bitwise_and(mask_foreground, mask_contour)

        # Apply erosion and dilation to remove noise
        kernel = np.ones((3, 3), np.uint8)
        kernelbig = np.ones((5, 5), np.uint8)
        mask_foreground = cv2.dilate(mask_foreground, kernel, iterations=1)
        mask_foreground = cv2.erode(mask_foreground, kernel, iterations=1)
        mask_foreground = cv2.erode(mask_foreground, kernelbig, iterations=1)

        frame = cv2.bitwise_and(frame, frame, mask=mask_foreground)

        # Remove the most common RGB value from the frame
        frame[frame == most_common_rgb_value_cam1] = 0
        frame[frame == most_common_rgb_value_cam2] = 0
        frame[frame == most_common_rgb_value_cam3] = 0
        frame[frame == most_common_rgb_value_cam4] = 0

        # Remove similar RGB values from the frame
        # for similar_rgb_value in similar_rgb_cam1:
        #     frame[frame == similar_rgb_value] = 0
    

        output_colour.append(frame)

        output_mask.append(mask_foreground)

    return output_colour, output_mask


if __name__ == "__main__":
    fps_background = ["./data/cam1/background.avi", "./data/cam2/background.avi",
                      "./data/cam3/background.avi", "./data/cam4/background.avi"]
    fps_foreground = ["./data/cam1/video.avi", "./data/cam2/video.avi",
                      "./data/cam3/video.avi", "./data/cam4/video.avi"]

    for c, fp_video in enumerate(fps_background):
        frames_background = read_frames(fp_video)
        if frames_background is None:
            print(f"Could not read frames from {fp_video}")
            continue

        frames_foreground = read_frames(fps_foreground[c], stop_after=10*50)
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
