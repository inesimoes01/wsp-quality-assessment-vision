import cv2
import numpy as np
import copy
import os
from matplotlib import pyplot as plt 
import sys 
sys.path.insert(0, 'src/common')
import config as config
import Util

class Distortion:
    def __init__(self, image, image_color, filename, save_photo):
        self.noPaper = False
        
        self.image = copy.copy(image)
        # detect the contour of the rectangle
        self.largest_contour = self.detect_rectangle_alternative(image_color)
        cv2.drawContours(self.image, [self.largest_contour], -1, (0, 0, 0), 5)

        # remove distortion from the image
        maxWidth, maxHeight = self.calculate_points(filename)

        if self.noPaper: return

        matrix = cv2.getPerspectiveTransform(self.input_pts, self.output_pts)
        self.undistorted_image = cv2.warpPerspective(image_color, matrix, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
   
        # save undistorted image
        if (save_photo): self.save_undistorted_image(filename)
    

        
    def calculate_points(self, filename):
        approx = cv2.approxPolyDP(self.largest_contour, 0.009 * cv2.arcLength(self.largest_contour, True), closed=True) 
        if len(approx) > 5: 
            print("Could not find the paper in image " + filename)
            self.noPaper = True
            
        # order corners
        approx = sorted(approx, key=lambda x: x[0][0] + x[0][1])
        top_left = approx[0][0]
        top_right = approx[1][0]
        bottom_left = approx[2][0]
        bottom_right = approx[3][0]

        # L2 norm
        width_AD = np.sqrt(((top_left[0] - top_right[0]) ** 2) + ((top_left[1] - top_right[1]) ** 2))
        width_BC = np.sqrt(((bottom_left[0] - bottom_right[0]) ** 2) + ((bottom_left[1] - bottom_right[1]) ** 2))
        maxWidth = max(int(width_AD), int(width_BC))
       
        height_AB = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
        height_CD = np.sqrt(((bottom_right[0] - top_right[0]) ** 2) + ((bottom_right[1] - top_right[1]) ** 2))
        maxHeight = max(int(height_AB), int(height_CD))
        
        self.input_pts = np.float32([top_left, bottom_left, bottom_right, top_right])
        self.output_pts = np.float32([[0, 0],
                                [0, maxHeight + 1],
                                [maxWidth +  1, maxHeight + 1],
                                [maxWidth + 1, 0]])
        #print("AFTER " + filename + " " + str(maxWidth) + " " + str(maxHeight))
        
        return maxWidth, maxHeight

    def detect_rectangle(self, image, image_color):
        blur = cv2.GaussianBlur(image, (9, 9), 0)
        #_, threshold_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, threshold_image = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY)
        kernel = np.ones((10, 10), np.uint8)
        dilation = cv2.dilate(threshold_image, kernel, iterations = 1)
        edges = cv2.Canny(dilation, 127, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        
        cv2.drawContours(image_color, contours, -1, (255, 0, 0), 4)
        cv2.drawContours(image_color, [contours[0]], -1, (255, 255, 0), 4)

        # hull = []
        # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for contour in contours:
        #     hull.append(cv2.convexHull(contour, False))

        # hull = sorted(hull, key=lambda x: cv2.contourArea(x), reverse=True)
        # # approx = cv2.approxPolyDP(hull[0], 0.009 * cv2.arcLength(hull[0], True), closed=True) 
        
        # cv2.drawContours(image_color, [approx], 0, (0, 0, 255), 5) 

        plt.imshow(image_color)
        plt.show()

        
        # linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=30)
        # if linesP is not None:
        #     for points in linesP:
        #         x1, y1, x2, y2 = points[0]
        #         cv2.line(image_color, (x1, y1),(x2, y2), (255, 0, 0), 5)

        # grouped_vertical_lines = self.group_lines_by_position(linesP, axis=0)
        # self.draw_grouped_lines(image_color, grouped_vertical_lines, axis=0)

        # # Group and draw lines by position for horizontal lines (axis=1)
        # grouped_horizontal_lines = self.group_lines_by_position(linesP, axis=1)
        # self.draw_grouped_lines(image_color, grouped_horizontal_lines, axis=1)

        # plt.imshow(image_color)
        # plt.show()

        #
        # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        # hull = []
        # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.floodFill(edges, None, (0, 0), 255)
        # plt.imshow(image)
        # plt.show()
        
        # # contour = contours[0]
        # # mask = np.zeros(image.shape, dtype=np.uint8)
        # # cv2.drawContours(mask, [contour], -1, 255, -1)
        # # mean_val = cv2.mean(image, mask=mask)[0]


        # # for contour in contours:
        # #     hull.append(cv2.convexHull(contour, False))
        # #     #cv2.drawContours(image, hull, -1, (255, 255, 255), 5)

        # cv2.drawContours(image_color, contours, -1, (255, 255, 255), 1)
        
        # plt.imshow(image_color)
        # plt.show()
        
        # when paper has almost no droplets, it is preferable to not apply morphology so not to delete the contour
        # if (len(contours) > 50):
        #     kernel_value = 20
        #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_value, kernel_value))
        #     edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=10)     # removing noise inside the rectangle
        #     kernel_value = 3
        #     edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)                     # removing noise outside the rectangle
        # else:            
        #     kernel = np.ones((5,5),np.uint8)
        #     edges = cv2.dilate(edges, kernel)
        #     edges = cv2.dilate(edges, kernel)
        #     edges = cv2.erode(edges, kernel)

        #     kernel_value = 3
        #     edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)                     # removing noise outside the rectangle
            
        # hull = []
        # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for contour in contours:
        #     hull.append(cv2.convexHull(contour, False))
        #     cv2.drawContours(image, hull, -1, (255, 0, 0), 10)

        # # only return the biggest contour
        # hull = sorted(hull, key=lambda x: cv2.contourArea(x), reverse=True)
        # return hull[0]

    
    def group_lines_by_position(self, lines, axis, threshold=5):
        grouped_lines = []
        lines = sorted(lines, key=lambda x: x[0][axis])
        current_group = [lines[0]]

        for line in lines[1:]:
            if abs(line[0][axis] - current_group[-1][0][axis]) < threshold:
                current_group.append(line)
            else:
                grouped_lines.append(current_group)
                current_group = [line]

        grouped_lines.append(current_group)
        return grouped_lines

    def draw_grouped_lines(self, image, grouped_lines, axis=0):
        for group in grouped_lines:
            coords = [line[0] for line in group]
            coords = sorted(coords, key=lambda x: x[1 - axis])
            start_point = tuple(coords[0][0:2])
            end_point = tuple(coords[-1][0:2])
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)

    def detect_rectangle_alternative(self, image):
    
        edges = cv2.GaussianBlur(image, (5, 5), 3, 3)

        # find the most present color
        histogram = cv2.calcHist([edges], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        max_count = np.amax(histogram)
        most_present_color = np.unravel_index(np.argmax(histogram), histogram.shape)
        most_present_color = tuple(int(c) for c in most_present_color)
        
        # remove colors based on upper and lower bounds
        tolerance = 100 
        lower_bound = np.array([max(0, c - tolerance) for c in most_present_color])
        upper_bound = np.array([min(255, c + tolerance) for c in most_present_color])
        mask = cv2.inRange(edges, lower_bound, upper_bound)
        result = cv2.bitwise_and(edges, edges, mask=cv2.bitwise_not(mask))

        # threshold image
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # find contours        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        hull = []
        #contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            hull.append(cv2.convexHull(contour, False))
            cv2.drawContours(image, hull, -1, (255, 0, 0), 5)
        
        # plt.imshow(image)
        # plt.show()
        return hull[0]

    def save_undistorted_image(self, filename):
        cv2.imwrite(os.path.join(config.DATA_REAL_RAW_DIR, filename + '.png'), self.undistorted_image);


# im2 = Distortion("images\\inesc_dataset\\1_V1_A1.jpg").undistorted_image
#cv2.imwrite("test1.png", im2)
# im3 = Distortion("images\\inesc_dataset\\1_V1_A2.jpg").undistorted_image
# im1 = Distortion("images\\real_images\\field1.jpg").undistorted_image
# plotThreeImages(im1, im2, im3)


        # # epsilon = 0.05 * cv2.arcLength(self.largest_contour, True)
        # # approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # if len(approx) == 4:
        #     src_pts = approx.reshape(4, 2)
        # else:
        #     print("Cannot find four corners or too many corners.")
        #     exit()
        

        # rect = cv2.minAreaRect(largest_contour)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)

        # x, y, w, h = cv2.boundingRect(largest_contour)
        # dest_pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype='float32')

        # Define the destination points for the perspective transformation
        # dest_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')
