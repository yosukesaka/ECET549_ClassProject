"""Feature matching and homography example using AKAZE.

This script is intended as a demonstration for a graduate-level
computer vision course. It loads baseline template images and a
captured scene image (from CoppeliaSim or disk), detects and matches
local features using AKAZE, filters matches with Lowe's ratio test,
estimates a homography with RANSAC, and visualizes/saves the results.

Files produced:
- <out_prefix>_matches.png : visualization of matched keypoints
- <out_prefix>_detected.png: scene image annotated with projected template corners

Typical usage: run the script from the directory that contains the
baseline images. This file is written to be readable and easily
modified for experimentation with different detectors, matchers, and
RANSAC thresholds.
"""

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import cv2
import time
import os
import sys


# def capture_image_from_sim(save_path='vision_sensor_0.png'):
#     try:
#         client = RemoteAPIClient('localhost', 23000)
#         sim = client.require('sim')
#         sensor0 = sim.getObject('/visionSensor')
#         sim.startSimulation()
#         time.sleep(0.5)
#         image0, res0 = sim.getVisionSensorImg(sensor0)
#         sim.stopSimulation()
#         img0 = np.frombuffer(image0, dtype=np.uint8).reshape((res0[1], res0[0], 3))
#         img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(save_path, img0)
#         print(f"Captured image from CoppeliaSim and saved to {save_path}")
#         return img0
#     except Exception as e:
#         print(f"Could not capture from CoppeliaSim: {e}")
#         return None


def load_image_safe(path):
    """Safely load an image from disk.

    Parameters
    - path: str or None
        Path to the image file. If None, returns None immediately.

    Returns
    - img: numpy.ndarray or None
        The loaded BGR image, or None if loading failed.

    Notes
    - Uses OpenCV's `imread`, which returns None on failure. This
      helper centralizes the check and prints a helpful message.
    """
    if path is None:
        return None
    img = cv2.imread(path)
    if img is None:
        print(f"Failed to read image: {path}")
    return img


def detect_and_match_and_draw(
    img_template,
    img_scene,
    img_template_color,
    img_scene_color,
    detector=None,
    ratio=0.9,
    show=False,
    out_prefix='matching',
    template_color=None,
    scene_color=None,
    norm=cv2.NORM_HAMMING,
    ransac_thresh=5.0,
):
    """Detect features, match them, estimate homography, and draw results.

    Parameters
    - img_template: grayscale numpy.ndarray
        Template image used for feature detection/matching (grayscale).
    - img_scene: grayscale numpy.ndarray
        Scene image to search for the template (grayscale).
    - img_template_color: BGR numpy.ndarray
        Color version of the template image used for visualization.
    - img_scene_color: BGR numpy.ndarray
        Color version of the scene image used for visualization.
    - detector: cv2.Feature2D or None
        OpenCV feature detector/extractor (e.g., AKAZE). If None,
        an AKAZE instance is created.
    - ratio: float
        Lowe's ratio threshold for filtering matches from knnMatch.
    - show: bool
        If True, display results in GUI windows.
    - out_prefix: str
        Prefix used when saving output images.
    - norm: OpenCV norm type
        Distance metric used by BFMatcher (default is Hamming for binary descriptors).
    - ransac_thresh: float
        Reprojection threshold (in pixels) used by RANSAC when computing homography.

    Returns
    - img_matches: BGR numpy.ndarray or None
        Visualization of matched keypoints (saved to disk as well).
    - img_draw: BGR numpy.ndarray or None
        Scene image annotated with the projected template corners.
    """

    # Create a default AKAZE detector if none provided
    if detector is None:
        detector = cv2.AKAZE_create()

    # Detect keypoints and compute descriptors for both images
    kp1, des1 = detector.detectAndCompute(img_template, None)
    kp2, des2 = detector.detectAndCompute(img_scene, None)

    # If either descriptor set is empty, matching cannot proceed
    if des1 is None or des2 is None:
        print('No descriptors found for one of the images.')
        return None, None

    # Brute-force matcher with specified norm (Hamming for binary descriptors)
    bf = cv2.BFMatcher(norm)
    # Use k-NN matching with k=2 for applying Lowe's ratio test
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test to retain good matches
    good = []
    for pair in matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    print(f"Total good matches: {len(good)}")

    # Prepare color copies for drawing (do not modify originals)
    img_template_bgr = img_template_color.copy()
    img_scene_bgr = img_scene_color.copy()

    # Visual copy of scene that we will annotate with the projected template
    img_draw = img_scene_bgr.copy()
    matchesMask = None

    # Need at least 4 matches to compute a homography
    if len(good) >= 4:
        # Build source and destination point arrays for homography estimation
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Estimate homography using RANSAC to be robust to outliers
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        if H is not None:
            # Project the template corners into the scene and draw them
            h, w = img_template.shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            projected = cv2.perspectiveTransform(corners, H)
            cv2.polylines(img_draw, [np.int32(projected)], True, (0, 255, 0), 3)
            matchesMask = mask.ravel().tolist() if mask is not None else None
            # mask contains 1 for inliers, sum(mask) gives number of inliers
            print(f"Homography found with {int(np.sum(mask))} inliers")
        else:
            print('Homography could not be computed')
    else:
        print('Not enough matches for homography')

    # Create a composite visualization of matches. Flags=2 draws only keypoints and lines.
    img_matches = cv2.drawMatches(
        img_template_bgr, kp1, img_scene_bgr, kp2, good, None, matchesMask=matchesMask, flags=2
    )

    # Save outputs to disk. OpenCV expects BGR ordering for imwrite.
    out_matches = f"{out_prefix}_matches.png"
    out_detect = f"{out_prefix}_detected.png"
    cv2.imwrite(out_matches, img_matches)
    cv2.imwrite(out_detect, img_draw)
    print(f"Saved {out_matches} and {out_detect}")

    # Optionally show results in GUI windows (blocks until key press)
    if show:
        cv2.imshow('matches', img_matches)
        cv2.imshow('detection', img_draw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img_matches, img_draw


def main():
    """Main function to load images and run matching for two templates.

    The script expects three files in the same directory as this script:
    - Baseline_hanger.png
    - Baseline_connector.png
    - vision_sensor_1.png (captured scene)

    The code below loads the images, converts to grayscale for
    descriptor extraction, creates an AKAZE detector instance, and
    runs the `detect_and_match_and_draw` routine for both templates.
    """

    base_dir = os.path.dirname(__file__)

    # Baseline images (expected to be located next to this script)
    path_hanger = os.path.join(base_dir, 'Baseline_hanger.png')
    path_connector = os.path.join(base_dir, 'Baseline_connector.png')
    path_capture = os.path.join(base_dir, 'vision_sensor_1.png')

    # Load images safely (prints an error and returns None if missing)
    img_hanger = load_image_safe(path_hanger)
    img_connector = load_image_safe(path_connector)
    img_scene = load_image_safe(path_capture)

    # If you prefer to capture from CoppeliaSim at runtime, uncomment
    # the capture code below. It uses the commented helper at the top
    # of this file (capture_image_from_sim) to request a snapshot from
    # the simulator.
    # img_scene = capture_image_from_sim(save_path=path_capture)
    # if img_scene is None:
    #     img_scene = load_image_safe(path_capture)

    # Basic sanity check; exit if required images are missing
    if img_hanger is None or img_connector is None or img_scene is None:
        print('One or more required images are missing. Exiting.')
        sys.exit(1)

    print(f"Images loaded: hanger={img_hanger.shape}, connector={img_connector.shape}, scene={img_scene.shape}")

    # Convert to grayscale for feature detection / descriptor extraction
    img_hanger_gray = cv2.cvtColor(img_hanger, cv2.COLOR_BGR2GRAY)
    img_connector_gray = cv2.cvtColor(img_connector, cv2.COLOR_BGR2GRAY)
    img_scene_gray = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)

    # Create a shared detector instance (AKAZE) and reuse for both templates
    detector = cv2.AKAZE_create()

    # Run matching for hanger template: grayscale for detection, color for visualization
    detect_and_match_and_draw(
        img_hanger_gray,
        img_scene_gray,
        img_hanger,
        img_scene,
        detector=detector,
        ratio=0.7,
        show=False,
        out_prefix='hanger_result',
        norm=cv2.NORM_HAMMING,
        ransac_thresh=4.0,
    )

    # Run matching for connector template with the same parameters
    detect_and_match_and_draw(
        img_connector_gray,
        img_scene_gray,
        img_connector,
        img_scene,
        detector=detector,
        ratio=0.7,
        show=False,
        out_prefix='connector_result',
        norm=cv2.NORM_HAMMING,
        ransac_thresh=4.0,
    )


if __name__ == '__main__':
    main()