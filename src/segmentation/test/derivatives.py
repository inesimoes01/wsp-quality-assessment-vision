import numpy as np
import cv2
import matplotlib.pyplot as plt

# Create an image with a contour
img = np.zeros((500, 500), dtype=np.uint8)
cv2.drawContours(img, [np.array([[50, 300], [200, 50], [350, 300], [150, 450]])], -1, 255, -1)  # Draw a simple contour

# Find contours
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = contours[0]

# Find the convex hull
hull = cv2.convexHull(contour, returnPoints=False)

# Find convexity defects
defects = cv2.convexityDefects(contour, hull)

# Create a blank image to draw points
output_img = np.zeros_like(img)

# Collect the convexity defect points
defect_points = []

# Iterate over the defects
if defects is not None:
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        defect_points.append(far)
        cv2.circle(output_img, far, 5, (255, 255, 255), -1)  # Draw defect points
        cv2.line(output_img, start, end, (255, 255, 255), 1)  # Draw hull edges
        cv2.circle(output_img, start, 5, (255, 255, 255), -1)  # Draw start points
        cv2.circle(output_img, end, 5, (255, 255, 255), -1)  # Draw end points

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image with Contour")
plt.imshow(img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Convexity Defects and Hull Points")
plt.imshow(output_img, cmap='gray')

plt.show()


# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt

# # Load or create an image with a contour
# # For demonstration, we'll create a binary image with a simple contour
# img = np.zeros((100, 100), dtype=np.uint8)
# cv.circle(img, (50, 50), 30, 255, 1)  # Draw a circle as the contour

# contours, _ = cv2.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)


# hull = cv.Mat();
# defect = new cv.Mat();
# let cnt = contours.get(0);
# let lineColor = new cv.Scalar(255, 0, 0);
# let circleColor = new cv.Scalar(255, 255, 255);
# cv.convexHull(cnt, hull, false, false);
# cv.convexityDefects(cnt, hull, defect);
# for (let i = 0; i < defect.rows; ++i) {
#     let start = new cv.Point(cnt.data32S[defect.data32S[i * 4] * 2],
#                              cnt.data32S[defect.data32S[i * 4] * 2 + 1]);
#     let end = new cv.Point(cnt.data32S[defect.data32S[i * 4 + 1] * 2],
#                            cnt.data32S[defect.data32S[i * 4 + 1] * 2 + 1]);
#     let far = new cv.Point(cnt.data32S[defect.data32S[i * 4 + 2] * 2],
#                            cnt.data32S[defect.data32S[i * 4 + 2] * 2 + 1]);
#     cv.line(dst, start, end, lineColor, 2, cv.LINE_AA, 0);
#     cv.circle(dst, far, 3, circleColor, -1);
# }
# cv.imshow('canvasOutput', dst);
# src.delete(); dst.delete(); hierarchy.delete(); contours.delete(); hull.delete(); defect.delete();


# # Find contours
# contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# contour = contours[0].squeeze()  # Get the first contour and remove singleton dimensions

# # Calculate derivatives using central differences
# def calculate_derivatives(contour):
#     dx = np.gradient(contour[:, 0])
#     dy = np.gradient(contour[:, 1])
#     return dx, dy

# # Calculate first derivatives
# dx, dy = calculate_derivatives(contour)

# # Calculate second derivatives
# ddx = np.gradient(dx)
# ddy = np.gradient(dy)

# # Visualize the contour and its derivatives
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.plot(contour[:, 0], contour[:, 1], label='Contour')
# plt.quiver(contour[:, 0], contour[:, 1], dx, dy, color='r', angles='xy', scale_units='xy', scale=1, label='First Derivative')
# plt.title('Contour and First Derivative')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(contour[:, 0], contour[:, 1], label='Contour')
# plt.quiver(contour[:, 0], contour[:, 1], ddx, ddy, color='g', angles='xy', scale_units='xy', scale=1, label='Second Derivative')
# plt.title('Contour and Second Derivative')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()

# plt.tight_layout()
# plt.show()


# # import numpy as np
# # import cv2
# # import matplotlib.pyplot as plt
# # from scipy.interpolate import splprep, splev

# # # Load or create an image with a contour
# # img = np.zeros((100, 100), dtype=np.uint8)
# # cv2.circle(img, (50, 50), 30, 255, 1)  # Draw a circle as the contour

# # # Find contours
# # contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# # contour = contours[0].squeeze()  # Get the first contour and remove singleton dimensions

# # # Check if contour is open or closed
# # if np.all(contour[0] == contour[-1]):
# #     contour = contour[:-1]  # Remove the duplicate endpoint if the contour is closed

# # # Perform spline interpolation
# # tck, u = splprep([contour[:, 0], contour[:, 1]], s=0, per=True)

# # # Evaluate the spline and its derivatives
# # unew = np.linspace(0, 1, len(contour))
# # out = splev(unew, tck)
# # der1 = splev(unew, tck, der=1)
# # der2 = splev(unew, tck, der=2)

# # # Convert the results to numpy arrays for easier handling
# # contour_interp = np.array(out).T
# # dx, dy = np.array(der1)
# # ddx, ddy = np.array(der2)

# # # Visualize the contour and its derivatives
# # plt.figure(figsize=(12, 6))

# # plt.subplot(1, 2, 1)
# # plt.plot(contour[:, 0], contour[:, 1], label='Original Contour')
# # plt.plot(contour_interp[:, 0], contour_interp[:, 1], 'r--', label='Spline Interpolated Contour')
# # plt.quiver(contour_interp[:, 0], contour_interp[:, 1], dx, dy, color='g', angles='xy', scale_units='xy', scale=1, label='First Derivative')
# # plt.title('Contour and First Derivative')
# # plt.xlabel('X')
# # plt.ylabel('Y')
# # plt.legend()

# # plt.subplot(1, 2, 2)
# # plt.plot(contour[:, 0], contour[:, 1], label='Original Contour')
# # plt.plot(contour_interp[:, 0], contour_interp[:, 1], 'r--', label='Spline Interpolated Contour')
# # plt.quiver(contour_interp[:, 0], contour_interp[:, 1], ddx, ddy, color='b', angles='xy', scale_units='xy', scale=1, label='Second Derivative')
# # plt.title('Contour and Second Derivative')
# # plt.xlabel('X')
# # plt.ylabel('Y')
# # plt.legend()

# # plt.tight_layout()
# # plt.show()
