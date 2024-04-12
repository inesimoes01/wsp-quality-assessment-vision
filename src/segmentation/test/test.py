import cv2
import numpy as np
import copy    
from PIL import Image
from matplotlib import pyplot as plt 

edge_image = cv2.Canny(edges, 200, 255, 2)
    
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    
    cont_image = copy.copy(image)
    convex_hull = []
    for contour in contours:
        # shape of a tight fitting convex boundary of the contours
        convex_hull.append(cv2.convexHull(contour))
        cv2.drawContours(cont_image, convex_hull, -1, (255, 255, 0), 5)
    
    convex_hull = sorted(convex_hull, key=lambda x: cv2.contourArea(x), reverse=True)
    convex_image = copy.copy(image)
    #minRect = cv2.boundingRect(np.array(convex_hull))
    #cv2.drawContours(convex_image, minRect, -1, (255, 255, 0), 5)
    # for convex in convex_hull:
    #     area = cv2.contourArea(convex)
    #     if (area > 1000):
            
    #         cv2.drawContours(convex_image, convex, -1, (255, 255, 0), 5)
    plt.imshow(cont_image)
    plt.show()

    #print(cv2.contourArea(convex_hull))
 # Copy edges to the images that will display the results in BGR
        cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)

        lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
        
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
        
        
        linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
        
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

        plotTwoImages(cdst, cdstP, None, None)

def detect_rectangle2(filename):
        image = cv2.imread(filename)
        # Convert image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        edges = cv2.GaussianBlur(hsv, (5, 5), 2, 2)
        edge_image= cv2.Canny(edges, 50, 150)

        # remove the noise edges of the droplet
        kernel_value = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_value, kernel_value))
        edge_image = cv2.morphologyEx(edge_image, cv2.MORPH_CLOSE, kernel, iterations=10)  # removing noise inside the rectangle
        kernel_value = 5
        edge_image = cv2.morphologyEx(edge_image, cv2.MORPH_OPEN, kernel)   # removing noise outside the rectangle

        # Find contours on the mask
        contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area, aspect ratio, etc.
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            area = cv2.contourArea(contour)
            if aspect_ratio > 0.8 and aspect_ratio < 1.2 and area > 1000:  # Adjust these thresholds as per your requirement
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box around paper
        
        plt.imshow(image)
        plt.show()
def detect_rectangle1(filename):
        img = cv2.imread(filename)

        edges = cv2.GaussianBlur(img, (5, 5), 2, 2)
        edge_image= cv2.Canny(edges, 50, 150)


        # remove the noise edges of the droplet
        kernel_value = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_value, kernel_value))
        edge_image = cv2.morphologyEx(edge_image, cv2.MORPH_CLOSE, kernel, iterations=10)  # removing noise inside the rectangle
        kernel_value = 5
        edge_image = cv2.morphologyEx(edge_image, cv2.MORPH_OPEN, kernel)   # removing noise outside the rectangle


        gray = np.float32(edge_image)
        dst = cv2.cornerHarris(gray,2,3,0.04)
        
        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)
        
        # Threshold for an optimal value, it may vary depending on the image.
        img[dst>0.01*dst.max()]=[255,255,0]
        
        plt.imshow(img)
        plt.show()

def detect_rectangle(filename):
        image = cv2.imread(filename)
        edges = cv2.GaussianBlur(image, (5, 5), 10, 10)
        kernel = np.ones((20,20),np.uint8)
        edges = cv2.erode(edges, kernel)
        edges = cv2.dilate(edges, kernel)
        edges = cv2.dilate(edges, kernel)
        edges = cv2.erode(edges, kernel)

        edge_image= cv2.Canny(edges, 100, 150)
        plt.imshow(edge_image)
        plt.show()
    
        # remove the noise edges of the droplet
        # kernel_value = 3
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_value, kernel_value))
        # edge_image = cv2.morphologyEx(edge_image, cv2.MORPH_CLOSE, kernel, iterations=10)  # removing noise inside the rectangle
        # kernel_value = 2
        # edge_image = cv2.morphologyEx(edge_image, cv2.MORPH_OPEN, kernel)   # removing noise outside the rectangle

        # find contours and order them
        # RETR_EXTERNAL - retrieves all of the contours without establishing any hierarchical relationships.
        # CHAIN_APPROX_SIMPLE - compresses horizontal, vertical, and diagonal segments and leaves only their end points.
        contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        
        hull = []
    
        # calculate points for each contour
        for i in range(len(contours)):
            # creating convex hull object for each contour
            hull.append(cv2.convexHull(contours[i], False))
            cv2.drawContours(image, hull, i, (255, 255, 0), 5)
            #     for contour in contours: 
            # convexHull = cv2.convexHull(contour)
        #cv2.drawContours(image, contours, 0, (255, 255, 0), 5)
        # convexHull = cv2.convexHull(contour)
        # cv2.drawContours(image, [convexHull], 0, (255, 255, 0), 5)
 
        # # assuming the biggest contour will be the rectangle
        # rectangle = contours[0]
        # convexHull = cv2.convexHull(rectangle)
        
        # rect = cv2.minAreaRect(rectangle)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)

        # cv2.drawContours(image,[convexHull],0,(255,255,0),5)
        plt.imshow(image)
        plt.show()

        #return convexHull


image = cv2.imread(filename,  cv2.IMREAD_GRAYSCALE)
edges = cv2.GaussianBlur(image, (5, 5), 40, 40)
# kernel = np.ones((5,5),np.uint8)
# edges = cv2.erode(edges, kernel)
# edges = cv2.dilate(edges, kernel)
# edges = cv2.dilate(edges, kernel)
# edges = cv2.erode(edges, kernel)
edges = cv2.Canny(edges, 200, 255, 1, 3, True)

kernel_value = 20
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_value, kernel_value))
edge_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=10)  # removing noise inside the rectangle
kernel_value = 5
edge_image = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)   # removing noise outside the rectangle

hull = []
contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    hull.append(cv2.convexHull(contour, False))
    cv2.drawContours(image, hull, -1, (255, 255, 255), 5)
    plt.imshow(image)
    plt.show()
    #cv2.drawContours(image, [contour], -1, (255, 255, 0), 5)



# import cv2
# import numpy as np

# from matplotlib import pyplot as plt 

# EPSILON = 1E-5

# def maximum(number1, number2, number3):
#     return max(max(number1, number2), number3)

# def almost_equal(number1, number2):
#     return abs(number1 - number2) <= (EPSILON * maximum(1.0, abs(number1), abs(number2)))
# def angle(pt1, pt2, pt0):
#     dx1 = pt1[0] - pt0[0]
#     dy1 = pt1[1] - pt0[1]
#     dx2 = pt2[0] - pt0[0]
#     dy2 = pt2[1] - pt0[1]
#     return (dx1*dx2 + dy1*dy2) / np.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10)

# def line_intersection(a1, b1, a2, b2):
#     A1 = b1[1] - a1[1]
#     B1 = a1[0] - b1[0]
#     C1 = (a1[0] * A1) + (a1[1] * B1)

#     A2 = b2[1] - a2[1]
#     B2 = a2[0] - b2[0]
#     C2 = (a2[0] * A2) + (a2[1] * B2)

#     det = (A1 * B2) - (A2 * B1)

#     if not almost_equal(det, 0):
#         x = ((C1 * B2) - (C2 * B1)) / det
#         y = ((C2 * A1) - (C1 * A2)) / det
#         return x, y
#     return None

# def sort_corners(corners):
#     top, bot = [], []
#     center = np.array([0, 0], dtype=np.float64)

#     for corner in corners:
#         center += corner
#     center *= (1.0 / len(corners))

#     for corner in corners:
#         if corner[1] < center[1]:
#             top.append(corner)
#         else:
#             bot.append(corner)

#     if len(top) == 2 and len(bot) == 2:
#         tl = min(top, key=lambda x: x[0])
#         tr = max(top, key=lambda x: x[0])
#         bl = min(bot, key=lambda x: x[0])
#         br = max(bot, key=lambda x: x[0])

#         return [tl, tr, br, bl]

# def main(image):
#     showsteps = True
#     src = cv2.imread(image)
#     # if src is None:
#     #     src = np.full((400, 400, 3), (127, 127, 127), dtype=np.uint8)
#     #     cv2.rectangle(src, (20, 200), (170, 250), (0, 0, 255), 8)
#     #     cv2.rectangle(src, (200, 200), (250, 250), (0, 0, 255), 8)

#     src_copy = src.copy()

#     edges = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

#     edges = cv2.GaussianBlur(edges, (5, 5), 1.5, 1.5)
#     edges = cv2.erode(edges, None)
#     edges = cv2.dilate(edges, None)
#     edges = cv2.dilate(edges, None)
#     edges = cv2.erode(edges, None)
#     edges = cv2.Canny(edges, 100, 150)   
#     contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#     selected_points = []
#     for contour in contours:
#         minRect = cv2.boundingRect(contour)
#         perimeter = cv2.arcLength(contour, True) * 0.02
#         approx = cv2.approxPolyDP(contour, perimeter, True)
#         # print(approx, " " ,  cv2.contourArea(approx))
#         # print("pii")
#         if len(approx) == 4 and cv2.contourArea(approx) > 100 and cv2.isContourConvex(approx):
#             max_cosine = 0
#             for j in range(2, 5):
#                 cosine = np.abs(angle(approx[j % 4][0], approx[j - 2][0], approx[j - 1][0]))
#                 max_cosine = max(max_cosine, cosine)
#             if max_cosine < 0.3:
#                 contours.append(approx)
#             print("oi?")
#             cv2.drawContours(src_copy, [contour], -1, (255, 255, 255), 5)
#         # if minRect[2] > src.shape[1] / 100 or minRect[3] > src.shape[0] / 100:
#         #     #print("ahhhhhh")
#         #     selected_points.extend(contour)
#         #     hull = cv2.convexHull(contour)
            
#         #     if showsteps:
#         #         cv2.drawContours(src_copy, [contour], -1, (255, 255, 255), 3)

#     plt.imshow(src_copy)
#     plt.show()
    
#     # if showsteps:
#     #     cv2.imshow("Selected contours", src_copy)
#     #     cv2.waitKey(1)


#     hulls = [cv2.convexHull(np.array(contour)) for contour in contours]

#     rect_points = []
#     for hull in hulls:
#         rect = cv2.minAreaRect(hull)
#         box = cv2.boxPoints(rect)
#         corners = sort_corners(box)
#         rect_points.append(corners)

#     quad = np.zeros((int(np.linalg.norm(corners[1] - corners[2])), int(np.linalg.norm(corners[2] - corners[3])), 3), dtype=np.uint8)

#     quad_pts = np.array([[0, 0], [quad.shape[1], 0], [quad.shape[1], quad.shape[0]], [0, quad.shape[0]]], dtype=np.float32)

#     transmtx = cv2.getPerspectiveTransform(np.array(corners, dtype=np.float32), quad_pts)
#     quad = cv2.warpPerspective(src, transmtx, (quad.shape[1], quad.shape[0]))

#     # if showsteps:
#     #     src_copy = src.copy()
#     #     cv2.polylines(src_copy, [np.array(corners, dtype=np.int32)], True, (0, 0, 255), 3)
#     #     cv2.imshow("selected quadrilateral part", src_copy)

#     # cv2.imshow("Result Image", quad)
#     # cv2.waitKey(0)

# if __name__ == "__main__":
#     main('images\\inesc_dataset\\1_V1_A1.jpg')












# img_path = 'images\\inesc_dataset\\1_V1_A1.jpg'
# img = cv2.imread(img_path)
# height, width, _ = img.shape


# patternSize = (7, 4)
# objectPoints = np.zeros((patternSize[0] * patternSize[1], 3), np.float32)
# objectPoints[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)

# allCalibrationImageCorners = []
# allObjectPoints = []
# allImagePoints = []


# retval, corners = Distortion.findChessboardCorners(img, patternSize)

# if retval != 0:
#     allObjectPoints.append(objectPoints)
#     allImagePoints.append(corners)

#     cv2.drawChessboardCorners(img, patternSize, corners, retval)
#     allCalibrationImageCorners.append(img)
# else:
#     print("Couldn't find all corners in:", img_path)

# height, width, channels = cv2.imread(img_path[0]).shape

# retval, INTRINSIC_MATRIX, DISTORTION_COEFFS, rvecs, tvecs = cv2.calibrateCamera(allObjectPoints, allImagePoints, (width, height), None, None)

# for calibrationImageCorners in allCalibrationImageCorners:
#     undistortedCalibrationImageCorners = Distortion.calculateOptimalUndistortion(calibrationImageCorners, height, width, INTRINSIC_MATRIX, DISTORTION_COEFFS)
#     Distortion.plotTwoImages(calibrationImageCorners, undistortedCalibrationImageCorners, "Distorted", "Undistorted")

# print("Intrinsic Matrix:\n", INTRINSIC_MATRIX)
# print("Lens Distortion Coefficients:\n", DISTORTION_COEFFS)

# connectedComponentAreaMin = 10
# undistExternalImg = Distortion.calculateOptimalUndistortion(img, height, width,  )
