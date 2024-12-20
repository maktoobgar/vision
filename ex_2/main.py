import optparse
import cv2
import numpy as np
import kagglehub


def otsu(image_path: str, save: bool = False):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Set foreground pixels to white
    image[thresh == 255] = 0

    # Save the result
    if save:
        cv2.imwrite("otsu.jpg", image)
    cv2.imshow("otsu_image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def iterative(image_path, save: bool = False):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Initialize variables
    threshold = 128  # Initial threshold
    max_value = 255
    change = True

    while change:
        # Apply current threshold
        binary_image = (image > threshold).astype(np.uint8) * max_value

        # Calculate new threshold as the average of the means of the two classes
        foreground_mean = np.mean(image[binary_image == max_value])
        background_mean = np.mean(image[binary_image == 0])
        new_threshold = (foreground_mean + background_mean) / 2

        # Check for convergence
        change = abs(new_threshold - threshold) > 1e-5  # Tolerance level
        threshold = new_threshold

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > threshold:
                image[i][j] = 0

    # Save the result
    if save:
        cv2.imwrite("iterative.jpg", image)
    cv2.imshow("iterative", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    path = kagglehub.dataset_download("nimapourmoradi/car-plate-detection-yolov8")

    print("Path to dataset files:", path)
    parser = optparse.OptionParser(
        usage="main.py [--otsu] --image=IMAGE [--save|]",
        description="runs otsu or iterative binary threshold finding algorithm",
    )
    parser.add_option(
        "--otsu",
        action="store_true",
        dest="otsu",
        help="runs otsu algorithm",
        default=False,
    )
    parser.add_option(
        "--iterative",
        action="store_true",
        dest="iterative",
        help="runs iterative algorithm",
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

    (options, args) = parser.parse_args()

    if options.otsu and len(options.image) > 0:
        otsu(options.image, options.save)
    elif options.iterative and len(options.image) > 0:
        iterative(options.image, options.save)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
