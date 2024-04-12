import cv2
import numpy as np
from matplotlib import pyplot as plt 

dp = 1
accum_ratio = 1.3
min_dist = 1
p1 = 200
p2 = 23
minDiam = 1
maxDiam = 0
scalebar = 10
min_range = 0
max_range = 100
intervals = 10
rad_list =[]
detected_circles = []
dataForTable = {}
threshold = 0
pixel_distance =1

def autoDetectBin(resized_img, threshold,accum_ratio, min_dist, p1, p2, minDiam, maxDiam, pixel_distance):
    global result, img, table_data, rad_list, detected_circles

    img = resized_img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thres,binImg = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # Blur using 3 * 3 kernel.
    blurred = cv2.blur(binImg, (3, 3))

    minDist = int(min_dist*pixel_distance)
    minRadius = int(minDiam*pixel_distance/2)
    maxRadius = int(maxDiam*pixel_distance/2)

    if minDist < 1:
        minDist = 1
    if minRadius <1:
        minRadius =1
    if minRadius <1:
        minRadius =1
    print(minDist, " ", minRadius, " ", maxRadius)
    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(blurred, 
                    cv2.HOUGH_GRADIENT, dp = int(accum_ratio), minDist = minDist, 
                    param1 = int(p1), param2 = int(p2), minRadius = minRadius, maxRadius = maxRadius)
    


def processCircles(state, resized_img, pixel_distance, manual_list):
    global detected_circles, rad_list, img, result, bottom_10percentile, top_90percentile, new_name
    # Draw circles that are detected.
    
    img = resized_img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rad_list=[]

    if state == False:
        detected_circles = None

    result = '\n\n'
    try:
        if (detected_circles is None) and (len(manual_list) == 0):
            print("no")
            return '\nNo circles found!\n'

        elif len(manual_list) > 0 and (detected_circles is None):
            print("no ", len(manual_list))
            manual_list.sort()
            bottom_10percentile = int(len(manual_list)*0.1)
            top_90percentile = int(len(manual_list)*0.9)
            result += '# of circles found: ' + str(len(manual_list))
            rad_list = manual_list

        else:
            
            # Convert the circle parameters a, b and r to integers.
            detected_circles = np.uint16(np.around(detected_circles))
            
            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]

                # Draw the circumference of the circle.
                cv2.circle(img, (a, b), r, (0, 255, 0), 2)

                # Draw a small circle (of radius 1) to show the center.
                cv2.circle(img, (a, b), 1, (0, 0, 255), 2)
                

            # new_name = filename[:-4] + '_detected' + filename[-4:]
            # cv2.imwrite(new_name, img)


            #Loop to convert radius (pixel) values to diameter
            for x in range(detected_circles.shape[1]):
                diam = detected_circles[0,x,2]*2/pixel_distance    
                rad_list.append(round(diam,1))

            rad_list.sort()

            bottom_10percentile = int(len(rad_list)*0.1)
            top_90percentile = int(len(rad_list)*0.9)

            result += '# of circles found: ' + str(detected_circles.shape[1]) 

        result +='\nAvg diam. = ' + "%.1f"%np.average(rad_list) + 'um' 
        result +='\nD10 = '+ str(rad_list[bottom_10percentile])+'um'+'\nD50 = ' + "%.1f"%np.median(rad_list) + "um" 
        result +='\nD90 = '+ str(rad_list[top_90percentile])+'um'

    except IndexError:
        pass
    return result

image_diam = [0,0]

im = cv2.imread("images\\artificial_dataset\\outputs\\overlapped\\2024-04-08_0\\72.png")
autoDetectBin(im, threshold, accum_ratio, min_dist, p1, p2, minDiam, maxDiam, pixel_distance)
processCircles(True, im, pixel_distance, image_diam)

plt.imshow(img)
plt.show()