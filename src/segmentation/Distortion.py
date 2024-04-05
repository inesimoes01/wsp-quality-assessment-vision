import cv2
import numpy as np
import copy
import math

from PIL import Image
from matplotlib import pyplot as plt 


class Distortion:
    def __init__(self, filename):
        self.image = cv2.imread(filename,  cv2.IMREAD_GRAYSCALE)
        # detect the contour of the rectangle
        contour = self.detect_rectangle(filename)
        #cv2.drawContours(image, [contour], -1, (255, 255, 0), 5)

        # draws the bounding rectangle with minimun area by considering the rotation
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(self.image, [box], -1, (255, 255, 0), 5)
         
      
        # plt.imshow(image)
        # plt.show()
        # # original_width = 26
        # # original_height = 76
        # real_width_cm = 2.6
        # real_height_cm = 7.6

        # # Convert real-world dimensions to pixels using the resolution of the image
        # # Assuming resolution in dpi (dots per inch)
        # dpi = 756.32  # example resolution, replace it with your actual resolution
        # dpi_to_cm_conversion = 1 / 2.54  # Conversion factor from inches to cm
        # original_width = int(real_width_cm * dpi * dpi_to_cm_conversion)
        # original_height = int(real_height_cm * dpi * dpi_to_cm_conversion)

        # # Calculate the transformation matrix
        # dest_pts = np.array([[0, 0], [original_width - 1, 0], [original_width - 1, original_height - 1], [0, original_height - 1]], dtype='float32')
        # matrix = cv2.getPerspectiveTransform(box.astype('float32'), dest_pts)

        # # Apply the transformation
        # undistorted_image = cv2.warpPerspective(image, matrix, (original_width, original_height))

        # # self.plotTwoImages(image, undistorted_image, "ah", "ah")
        # # plt.show()
        # # # Save or display the undistorted image
        # # cv2.imshow('Undistorted Image', undistorted_image)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()
    



        # calibrationImageGray = cv2.cvtColor(calibrationImage, cv2.COLOR_BGR2GRAY)
        # retval, corners = cv2.findChessboardCorners(calibrationImageGray, patternSize)
        # print("")
        # if retval != 0:
        #     corners = cv2.cornerSubPix(calibrationImageGray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001))
        #     return retval, corners
        # else:
        #     return 0, None
    
    def detect_rectangle(self, filename):
        image = cv2.imread(filename,  cv2.IMREAD_GRAYSCALE)
        edges = cv2.GaussianBlur(image, (5, 5), 3, 3)

        edges = cv2.Canny(edges, 150, 200, 1, 3, True)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
        # when paper has almost no droplets, it is preferable to not apply morphology so not to delete the contour
        if (len(contours) > 50):
            kernel_value = 20
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_value, kernel_value))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=10)     # removing noise inside the rectangle
            kernel_value = 3
            edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)                     # removing noise outside the rectangle
        else:            
            kernel = np.ones((5,5),np.uint8)
            edges = cv2.dilate(edges, kernel)
            edges = cv2.dilate(edges, kernel)
            edges = cv2.erode(edges, kernel)

            kernel_value = 3
            edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)                     # removing noise outside the rectangle
            
        hull = []
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            hull.append(cv2.convexHull(contour, False))
            cv2.drawContours(image, hull, -1, (255, 255, 255), 5)
        
        # only return the biggest contour
        hull = sorted(hull, key=lambda x: cv2.contourArea(x), reverse=True)
        return hull[0]



def sort_corners(corners):
    top, bot = [], []
    center = np.array([0, 0], dtype=np.float64)

    for corner in corners:
        center += corner
    center *= (1.0 / len(corners))

    for corner in corners:
        if corner[1] < center[1]:
            top.append(corner)
        else:
            bot.append(corner)

    if len(top) == 2 and len(bot) == 2:
        tl = min(top, key=lambda x: x[0])
        tr = max(top, key=lambda x: x[0])
        bl = min(bot, key=lambda x: x[0])
        br = max(bot, key=lambda x: x[0])

        return [tl, tr, br, bl]  
       
def plotThreeImages(image1, image2, image3):
    # Create a side-by-side plot with titles
    plt.close('all')
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))

    axes[0].imshow(image1)
    #axes[0].axis('off')
    axes[0].set_xlabel("X (pixels)")
    axes[0].set_ylabel("Y (pixels)")

    axes[1].imshow(image2)
    #axes[1].axis('off')
    axes[1].set_xlabel("X (pixels)")
    axes[1].set_ylabel("Y (pixels)")
    
    axes[2].imshow(image3)
    #axes[1].axis('off')
    axes[2].set_xlabel("X (pixels)")
    axes[2].set_ylabel("Y (pixels)")


    plt.show()

 

im1 = Distortion("images\\inesc_dataset\\2_V1_A3.jpg").image
im2 = Distortion("images\\inesc_dataset\\2_V1_A1.jpg").image
im3 = Distortion("images\\inesc_dataset\\2_V1_A2.jpg").image
plotThreeImages(im1, im2, im3)