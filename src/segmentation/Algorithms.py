import cv2
import numpy as np
import sys
import copy
from matplotlib import pyplot as plt 

sys.path.insert(0, 'src')
from Droplet import *


class Algorithms:
    def __init__(self):
        image_path = 'images\\artificial_dataset\\outputs\\overlapped\\2024-03-22_0\\32.png'
        # image_path = 'images\\artificial_dataset\\image\\2024-03-25_0.png'
        self.original_image = cv2.imread(image_path)
        self.image = cv2.imread(image_path)
        
        # apply hough_tansform algorithm
        # self.hough_tansform()

        # apply ransac algorithm
        self.no_iterations = 50000
        self.radius_threshold = 16
        self.edge_points_threshold = 5
        self.ransac()

    def process_image(self):
        self.image_blur = cv2.GaussianBlur(self.image, (7, 7), 1.5)
        self.gray = cv2.cvtColor(self.image_blur, cv2.COLOR_BGR2GRAY)
        self.edges = cv2.Canny(self.gray, 50, 150)
        _, self.thresh = cv2.threshold(self.edges, 127, 255, cv2.THRESH_BINARY)
        self.contours, _ = cv2.findContours(self.edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.edge_points = [point[0] for contour in self.contours for point in contour]

    def hough_tansform(self):
        self.process_image()

        # Find circles using Hough transform
        circles = cv2.HoughCircles(self.edges, cv2.HOUGH_GRADIENT, dp=1.3, minDist=10, param1=150, param2=70, minRadius=0, maxRadius=0)

        if circles is not None:
            # Convert circle parameters to integer
            circles = np.uint16(np.around(circles))

            # Draw detected circles
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                cv2.circle(self.image, center, radius, (0, 255, 0), 2)

            # Display result
            cv2.imshow('Detected Circles', self.image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No circles detected in the image.")

    def least_square(self):
        return 
    
    def ransac(self):
        self.process_image()

        # run algorithm
        best_circles_1 = self.ransac_helper(self.edge_points)

        # annotate the image + remove circles detected
        thresh_1 = copy.copy(self.thresh)
        mask_1 = np.zeros_like(self.gray)
        image_to_annotate = copy.copy(self.original_image)

        for circle in best_circles_1:
            print("1st")
            cv2.circle(image_to_annotate, (int(circle.center_x), int(circle.center_y)), int(circle.radius), (0, 255, 0), 1)

            # remove pixels from image
            for i in range(int(circle.radius)):
                cv2.circle(mask_1, (int(circle.center_x), int(circle.center_y)), int(circle.radius)-i, (255), 2)

        mask_inv_1 = cv2.bitwise_not(mask_1)
        result_1 = cv2.bitwise_and(thresh_1, thresh_1, mask=mask_inv_1)

        # run algorithm again
        best_circles_2 = self.ransac_helper(self.edge_points)

        for circle in best_circles_2:
            print("2nd")
            cv2.circle(image_to_annotate, (int(circle.center_x), int(circle.center_y)), int(circle.radius), (255, 0, 0), 1)

        fig = plt.figure(figsize=(10, 7)) 
        fig.add_subplot(2, 1, 1)
        plt.imshow(image_to_annotate)
        fig.add_subplot(2, 1, 2)
        plt.imshow(result_1)
        # fig.add_subplot(2, 2, 3)
        # plt.imshow(result2)
        # fig.add_subplot(2, 2, 4)
        # plt.imshow(image3)
        plt.show() 
    
    def ransac_helper(self, edge_points):
        best_circles = []
        for _ in range(self.no_iterations):
            # Select three points at random among edge points
            indices = np.random.choice(len(edge_points), 3, replace=False)
            A, B, C = [np.array(edge_points[i]) for i in indices]

            # Calculate midpoints
            midpt_AB = (A + B) * 0.5
            midpt_BC = (B + C) * 0.5

            # Calculate slopes and intercepts
            slope_AB = (B[1] - A[1]) / (B[0] - A[0] + 0.000000001)
            intercept_AB = A[1] - slope_AB * A[0]
            slope_BC = (C[1] - B[1]) / (C[0] - B[0] + 0.000000001)
            intercept_BC = C[1] - slope_BC * C[0]

            # Calculate perpendicular slopes and intercepts
            slope_midptAB = -1.0 / slope_AB
            slope_midptBC = -1.0 / slope_BC
            intercept_midptAB = midpt_AB[1] - slope_midptAB * midpt_AB[0]
            intercept_midptBC = midpt_BC[1] - slope_midptBC * midpt_BC[0]

            # Calculate intersection of perpendiculars to find center of circle and radius
            centerX = (intercept_midptBC - intercept_midptAB) / (slope_midptAB - slope_midptBC)
            centerY = slope_midptAB * centerX + intercept_midptAB
            center = (centerX, centerY)
            radius = np.linalg.norm(center - A)
            circumference = 2.0 * np.pi * radius

            on_circle = []
            not_on_circle = []
            

            # Find edge points that fit on circle radius
            for i, point in enumerate(edge_points):
                distance_to_center = np.linalg.norm(np.array(point) - np.array(center))
                if abs(distance_to_center - radius) < self.radius_threshold:
                    on_circle.append(i)
                else:
                    not_on_circle.append(i)
        
            # If number of edge points more than circumference, we found a correct circle
            if len(on_circle) >= circumference:
                best_circles.append(Droplet(centerX, centerY, radius))

                # Remove edge points if circle found (only keep non-voting edge points)
                edge_points = [edge_points[i] for i in not_on_circle]

            # Stop iterations when there are not enough edge points
            if len(edge_points) < self.edge_points_threshold:
                break

        return best_circles

    def gradient_descent(self):
        return 




Algorithms()