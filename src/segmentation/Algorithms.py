# import cv2
# import numpy as np
# import sys
# import copy
# from matplotlib import pyplot as plt 

# import time


# sys.path.insert(0, 'src/common')
# from Droplet import *
# from Util import *
# from Distortion import *


# class Algorithms:
#     # def __init__(self):
        
#         # image_path = 'images\\artificial_dataset\\image\\2024-03-25_0.png'
#         # self.original_image = cv2.imread(image_path)
#         # self.image = cv2.imread(image_path)
        
#         # hough_tansform algorithm
#         # self.output = self.hough_transform_paper(image_path)

#         # # apply ransac algorithm
#         # self.no_iterations = 50000
#         # self.radius_threshold = 16
#         # self.edge_points_threshold = 5
#         # self.ransac()


#     def process_image(self, image):
#         image_blur = cv2.GaussianBlur(image, (7, 7), 1.5)
#         gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
#         edges = cv2.Canny(gray, 150, 200)

#         _, thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
#         contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         edge_points = [point[0] for contour in contours for point in contour]
#         return edges

#     def hough_tansform(self, file_path, image):
#         edges = self.process_image(image)

#         circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.3, minDist=1, param1=200, param2=23, minRadius=1, maxRadius=0)

#         if circles is not None:
#             # Convert circle parameters to integer
#             circles = np.uint16(np.around(circles))

#             # draw detected circles
#             for circle in circles[0, :]:
#                 center = (circle[0], circle[1])
#                 radius = circle[2]
#                 cv2.circle(image, center, radius, (0, 255, 0), 2)

#             cv2.imwrite(file_path, image)
#         else:
#             print("No circles detected in the image.")

#     def least_square(self):
#         return 
    
#     def ransac(self):
#         self.process_image()

#         # run algorithm
#         best_circles_1 = self.ransac_helper(self.edge_points)

#         # annotate the image + remove circles detected
#         thresh_1 = copy.copy(self.thresh)
#         mask_1 = np.zeros_like(self.gray)
#         image_to_annotate = copy.copy(self.original_image)

#         for circle in best_circles_1:
#             print("1st")
#             cv2.circle(image_to_annotate, (int(circle.center_x), int(circle.center_y)), int(circle.radius), (0, 255, 0), 1)

#             # remove pixels from image
#             for i in range(int(circle.radius)):
#                 cv2.circle(mask_1, (int(circle.center_x), int(circle.center_y)), int(circle.radius)-i, (255), 2)

#         mask_inv_1 = cv2.bitwise_not(mask_1)
#         result_1 = cv2.bitwise_and(thresh_1, thresh_1, mask=mask_inv_1)

#         # run algorithm again
#         best_circles_2 = self.ransac_helper(self.edge_points)

#         for circle in best_circles_2:
#             print("2nd")
#             cv2.circle(image_to_annotate, (int(circle.center_x), int(circle.center_y)), int(circle.radius), (255, 0, 0), 1)

#         fig = plt.figure(figsize=(10, 7)) 
#         fig.add_subplot(2, 1, 1)
#         plt.imshow(image_to_annotate)
#         fig.add_subplot(2, 1, 2)
#         plt.imshow(result_1)
#         # fig.add_subplot(2, 2, 3)
#         # plt.imshow(result2)
#         # fig.add_subplot(2, 2, 4)
#         # plt.imshow(image3)
#         plt.show() 
    
#     def ransac_helper(self, edge_points):
#         best_circles = []
#         for _ in range(self.no_iterations):
#             # Select three points at random among edge points
#             indices = np.random.choice(len(edge_points), 3, replace=False)
#             A, B, C = [np.array(edge_points[i]) for i in indices]

#             # Calculate midpoints
#             midpt_AB = (A + B) * 0.5
#             midpt_BC = (B + C) * 0.5

#             # Calculate slopes and intercepts
#             slope_AB = (B[1] - A[1]) / (B[0] - A[0] + 0.000000001)
#             intercept_AB = A[1] - slope_AB * A[0]
#             slope_BC = (C[1] - B[1]) / (C[0] - B[0] + 0.000000001)
#             intercept_BC = C[1] - slope_BC * C[0]

#             # Calculate perpendicular slopes and intercepts
#             slope_midptAB = -1.0 / slope_AB
#             slope_midptBC = -1.0 / slope_BC
#             intercept_midptAB = midpt_AB[1] - slope_midptAB * midpt_AB[0]
#             intercept_midptBC = midpt_BC[1] - slope_midptBC * midpt_BC[0]

#             # Calculate intersection of perpendiculars to find center of circle and radius
#             centerX = (intercept_midptBC - intercept_midptAB) / (slope_midptAB - slope_midptBC)
#             centerY = slope_midptAB * centerX + intercept_midptAB
#             center = (centerX, centerY)
#             radius = np.linalg.norm(center - A)
#             circumference = 2.0 * np.pi * radius

#             on_circle = []
#             not_on_circle = []
            

#             # Find edge points that fit on circle radius
#             for i, point in enumerate(edge_points):
#                 distance_to_center = np.linalg.norm(np.array(point) - np.array(center))
#                 if abs(distance_to_center - radius) < self.radius_threshold:
#                     on_circle.append(i)
#                 else:
#                     not_on_circle.append(i)
        
#             # If number of edge points more than circumference, we found a correct circle
#             if len(on_circle) >= circumference:
#                 best_circles.append(Droplet(centerX, centerY, radius))

#                 # Remove edge points if circle found (only keep non-voting edge points)
#                 edge_points = [edge_points[i] for i in not_on_circle]

#             # Stop iterations when there are not enough edge points
#             if len(edge_points) < self.edge_points_threshold:
#                 break

#         return best_circles

#     def gradient_descent(self):
#         return 
    
#     def hough_transform_paper(self, image):

#         original_image = cv2.imread(image, 1)
#         #gray_image = cv2.imread('Sample_Input.jpg',0)
#         #cv2.imshow('Original Image',original_image)

#         self.output = original_image.copy()

#         #Gaussian Blurring of Gray Image
#         blur_image = cv2.GaussianBlur(original_image,(3,3),0)
#         #cv2.imshow('Gaussian Blurred Image',blur_image)

#         #Using OpenCV Canny Edge detector to detect edges
#         edged_image = cv2.Canny(blur_image,75,150)
#         #cv2.imshow('Edged Image', edged_image)

#         height,width = edged_image.shape
#         radii = 100

#         acc_array = np.zeros(((height,width,radii)))

#         filter3D = np.zeros((30,30,radii))
#         filter3D[:,:,:]=1

#         start_time = time.time()

#         edges = np.where(edged_image==255)

#         for i in xrange(0,len(edges[0])):
#             x=edges[0][i]
#             y=edges[1][i]
#             for radius in range(20,55):
#                 self.fill_acc_array(x,y,radius, height, width, acc_array)
                    
#         i=0
#         j=0

#         while(i<height-30):
#             while(j<width-30):
#                 filter3D=acc_array[i:i+30,j:j+30,:]*filter3D
#                 max_pt = np.where(filter3D==filter3D.max())
#                 a = max_pt[0]       
#                 b = max_pt[1]
#                 c = max_pt[2]
#                 b=b+j
#                 a=a+i
#                 if(filter3D.max()>90):
#                     cv2.circle(self.output,(b,a),c,(0,255,0),2)
#                 j=j+30
#                 filter3D[:,:,:]=1
#             j=0
#             i=i+30

        
#         end_time = time.time()
#         time_taken = end_time - start_time
#         print ('Time taken for execution',time_taken)
#         plt.imshow(self.output)
#         plt.show()
#         return self.output

#     def fill_acc_array(self, x0,y0,radius, height, width, acc_array):
#         x = radius
#         y=0
#         decision = 1-x
        
#         while(y<x):
#             if(x + x0<height and y + y0<width):
#                 acc_array[ x + x0,y + y0,radius]+=1; # Octant 1
#             if(y + x0<height and x + y0<width):
#                 acc_array[ y + x0,x + y0,radius]+=1; # Octant 2
#             if(-x + x0<height and y + y0<width):
#                 acc_array[-x + x0,y + y0,radius]+=1; # Octant 4
#             if(-y + x0<height and x + y0<width):
#                 acc_array[-y + x0,x + y0,radius]+=1; # Octant 3
#             if(-x + x0<height and -y + y0<width):
#                 acc_array[-x + x0,-y + y0,radius]+=1; # Octant 5
#             if(-y + x0<height and -x + y0<width):
#                 acc_array[-y + x0,-x + y0,radius]+=1; # Octant 6
#             if(x + x0<height and -y + y0<width):
#                 acc_array[ x + x0,-y + y0,radius]+=1; # Octant 8
#             if(y + x0<height and -x + y0<width):
#                 acc_array[ y + x0,-x + y0,radius]+=1; # Octant 7
#             y+=1
#             if(decision<=0):
#                 decision += 2 * y + 1
#             else:
#                 x=x-1;
#                 decision += 2 * (y - x) + 1



# # im_array = []
# # i=0 
# # for file in os.listdir("images\\artificial_dataset\\outputs\\overlapped\\2024-04-08_0"):
# #     if (i == 36): break
# #     im_array.append(Algorithms("images\\artificial_dataset\\outputs\\overlapped\\2024-04-08_0\\" + file).output)
# #     i+=1
# # print(i)


# # plotALOTImages(im_array)
# # im1 = Algorithms("images\\artificial_dataset\\outputs\\overlapped\\2024-04-08_0\\302.png").image
# # im2 = Algorithms("images\\artificial_dataset\\outputs\\overlapped\\2024-04-08_0\\72.png").image
# # im3 = Algorithms("images\\artificial_dataset\\outputs\\overlapped\\2024-04-08_0\\279.png").image
# # plotThreeImages(im1, im2, im3)