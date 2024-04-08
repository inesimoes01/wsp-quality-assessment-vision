import cv2
import numpy as np
import matplotlib.pyplot as plt

def createCountournImage(imagePath):
    image = cv2.imread(imagePath)

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calculate histogram of the image
    histogram = cv2.calcHist([image_rgb], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

    # Find the most present color
    max_count = np.amax(histogram)
    most_present_color = np.unravel_index(np.argmax(histogram), histogram.shape)

    # Convert the color to uint8
    most_present_color = tuple(int(c) for c in most_present_color)

    # Define upper and lower bounds for colors to be removed
    tolerance = 100  # Adjust this value to include similar shades of the color
    lower_bound = np.array([max(0, c - tolerance) for c in most_present_color])
    upper_bound = np.array([min(255, c + tolerance) for c in most_present_color])

    # Set pixels with colors near the most present color to white
    mask = cv2.inRange(image_rgb, lower_bound, upper_bound)
    result = cv2.bitwise_and(image_rgb, image_rgb, mask=cv2.bitwise_not(mask))

    # Convert the result image to grayscale
    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

    # Threshold the grayscale image
    ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Create a black background image
    black_background = np.zeros_like(image_rgb)

    # Draw the biggest contour on the black background image
    x, y, w, h = cv2.boundingRect(contours[0])
    cv2.rectangle(black_background, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Initial image on the left
    axs[0].imshow(image_rgb)
    axs[0].set_title('Initial Image')
    axs[0].axis('off')

    # Image with rectangle outline on the right
    axs[1].imshow(black_background)
    axs[1].set_title('Initial Image with Rectangle Outline')
    axs[1].axis('off')

    plt.show()

createCountournImage("images\\inesc_dataset\\2_V1_A3.jpg")
createCountournImage("images\\inesc_dataset\\2_V1_A1.jpg")
createCountournImage("images\\inesc_dataset\\2_V1_A2.jpg")