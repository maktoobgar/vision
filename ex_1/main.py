import cv2
import os
import imutils
import optparse
import matplotlib.pyplot as plt
from skimage import io, color, feature
from skimage.transform import resize
from skimage.util import img_as_ubyte
from skimage.measure import ransac
import numpy as np
import matplotlib.pyplot as plt


def surf(image: str, second_image: str, save: bool = False):
    # Read and preprocess images
    image1 = img_as_ubyte(color.rgb2gray(io.imread(image)))
    image2 = img_as_ubyte(color.rgb2gray(io.imread(second_image)))

    # Resize for efficiency (optional)
    image1 = resize(image1, (image1.shape[0] // 2, image1.shape[1] // 2))
    image2 = resize(image2, (image2.shape[0] // 2, image2.shape[1] // 2))

    # Detect and extract features using ORB
    orb = feature.ORB(n_keypoints=500)
    orb.detect_and_extract(image1)
    kp1, ds1 = orb.keypoints, orb.descriptors

    orb.detect_and_extract(image2)
    kp2, ds2 = orb.keypoints, orb.descriptors

    # Match descriptors
    matches = feature.match_descriptors(ds1, ds2, cross_check=True)

    # Convert keypoints to cv2 KeyPoint objects
    keypoints1_cv = [cv2.KeyPoint(p[1], p[0], 1) for p in kp1]
    keypoints2_cv = [cv2.KeyPoint(p[1], p[0], 1) for p in kp2]

    # Create DMatch objects for matched points
    matches_cv = [cv2.DMatch(i, j, 0) for i, j in matches]

    # Draw matches using cv2
    matched_image = cv2.drawMatches(
        (image1 * 255).astype(np.uint8),
        keypoints1_cv,
        (image2 * 255).astype(np.uint8),
        keypoints2_cv,
        matches_cv,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    keypoints_image = cv2.drawKeypoints(
        image=(image1 * 255).astype(np.uint8),
        outImage=(image1 * 255).astype(np.uint8),
        keypoints=keypoints1_cv,
        flags=4,
        color=(0, 255, 0),
    )

    cv2.imwrite("surf_matched_image.jpg", matched_image)
    cv2.imwrite("surf_keypoints.jpg", keypoints_image)

    # Stitch images using homography
    if len(matches) > 10:  # Enough matches to find a homography
        src_pts = np.float32([kp1[m[0]] for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m[1]] for m in matches]).reshape(-1, 1, 2)

        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

        # Warp the first image to the second image's plane
        height, width = image2.shape
        stitched_image = cv2.warpPerspective(
            (image1 * 255).astype(np.uint8), M, (width, height)
        )

        # Optionally blend the images (simple blending)
        result = cv2.addWeighted(
            stitched_image, 0.5, (image2 * 255).astype(np.uint8), 0.5, 0
        )

        # Display stitched result
        cv2.imshow("Stitched Image", result)
        if save:
            cv2.imwrite("surf_stiched_image.jpg", result)
    else:
        print("Not enough matches to find a homography.")

    if save:
        cv2.imwrite("surf_matched_image.jpg", matched_image)
        cv2.imwrite("surf_keypoints.jpg", keypoints_image)
    cv2.imshow("Matched Image", matched_image)
    cv2.imshow("Keypoints", keypoints_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sift(image: str, second_image: str, save: bool = False):
    image1 = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(second_image, cv2.IMREAD_GRAYSCALE)
    image1 = imutils.resize(image1, width=image1.shape[1] // 2)
    image2 = imutils.resize(image2, width=image2.shape[1] // 2)

    sift = cv2.SIFT_create()
    kp1, ds1 = sift.detectAndCompute(image1, None)
    kp2, ds2 = sift.detectAndCompute(image2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(ds1, ds2)
    matches = sorted(matches, key=lambda x: x.distance)

    matched_image = cv2.drawMatches(
        image1, kp1, image2, kp2, matches[:40], None, flags=2
    )
    keypoints = cv2.drawKeypoints(
        image=image1, outImage=image1, keypoints=kp1, flags=4, color=(0, 255, 0)
    )

    # Find homography and stitch the images
    if len(matches) > 10:  # Need a minimum number of matches to find a homography
        # src_pts = np.float32([kp1[m[0]] for m in matches]).reshape(-1, 1, 2)
        # dst_pts = np.float32([kp2[m[1]] for m in matches]).reshape(-1, 1, 2)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

        # Warp the first image to the second image's plane
        height, width = image2.shape
        stitched_image = cv2.warpPerspective(
            (image1).astype(np.uint8), M, (width, height)
        )

        # Optionally blend the images (simple blending)
        result = cv2.addWeighted(stitched_image, 0.5, (image2).astype(np.uint8), 0.5, 0)

        # Display stitched result
        cv2.imshow("Stitched Image", result)
        if save:
            cv2.imwrite("sift_stiched_image.jpg", result)
    else:
        print("Not enough matches to find a homography.")

    if save:
        cv2.imwrite("sift_matched_image.jpg", matched_image)
        cv2.imwrite("sift_keypoints.jpg", keypoints)
    cv2.imshow("Matched Image", matched_image)
    cv2.imshow("Keypoints", keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def orb(image: str, second_image: str, save: bool = False):
    image1 = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(second_image, cv2.IMREAD_GRAYSCALE)

    # Resize images for processing
    image1 = imutils.resize(image1, width=image1.shape[1] // 2)
    image2 = imutils.resize(image2, width=image2.shape[1] // 2)

    # Detect keypoints and descriptors using ORB
    orb = cv2.ORB_create()
    kp1, ds1 = orb.detectAndCompute(image1, None)
    kp2, ds2 = orb.detectAndCompute(image2, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(ds1, ds2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw matches for visualization
    matched_image = cv2.drawMatches(
        image1, kp1, image2, kp2, matches[:40], image2, flags=2
    )
    keypoints = cv2.drawKeypoints(
        image=image1, outImage=image1, keypoints=kp1, flags=4, color=(0, 255, 0)
    )

    # Find homography and stitch the images
    if len(matches) > 10:  # Need a minimum number of matches to find a homography
        # src_pts = np.float32([kp1[m[0]] for m in matches]).reshape(-1, 1, 2)
        # dst_pts = np.float32([kp2[m[1]] for m in matches]).reshape(-1, 1, 2)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

        # Warp the first image to the second image's plane
        height, width = image2.shape
        stitched_image = cv2.warpPerspective(
            (image1).astype(np.uint8), M, (width, height)
        )

        # Optionally blend the images (simple blending)
        result = cv2.addWeighted(stitched_image, 0.5, (image2).astype(np.uint8), 0.5, 0)

        # Display stitched result
        cv2.imshow("Stitched Image", result)
        if save:
            cv2.imwrite("orb_stiched_image.jpg", result)
    else:
        print("Not enough matches to find a homography.")

    # Display matched keypoints and keypoints on first image
    if save:
        cv2.imwrite("orb_matched_image.jpg", matched_image)
        cv2.imwrite("orb_keypoints.jpg", keypoints)
    cv2.imshow("Matched Image", matched_image)
    cv2.imshow("Keypoints", keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = optparse.OptionParser(description="runs sift, surf or orb algorithms")
    parser.add_option(
        "--sift",
        action="store_true",
        dest="sift",
        help="runs sift algorithm",
        default=False,
    )
    parser.add_option(
        "--surf",
        action="store_true",
        dest="surf",
        help="runs surf algorithm",
        default=False,
    )
    parser.add_option(
        "--orb",
        action="store_true",
        dest="orb",
        help="runs orb algorithm",
        default=False,
    )
    parser.add_option(
        "--save",
        action="store_true",
        dest="save",
        help="saves the result images",
        default=False,
    )
    parser.add_option(
        "--image",
        dest="image",
        type=str,
        default="",
        help="image relative address to use",
    )
    parser.add_option(
        "--second_image",
        dest="second_image",
        type=str,
        default="",
        help="second_image relative address to use",
    )

    (options, args) = parser.parse_args()

    if options.sift and len(options.image) > 0 and len(options.second_image) > 0:
        sift(options.image, options.second_image, options.save)
    elif options.surf and len(options.image) > 0 and len(options.second_image) > 0:
        surf(options.image, options.second_image, options.save)
    elif options.orb and len(options.image) > 0 and len(options.second_image) > 0:
        orb(options.image, options.second_image, options.save)
    else:
        print("invalid command")


if __name__ == "__main__":
    main()
