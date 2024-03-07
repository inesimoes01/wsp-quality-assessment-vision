import cv2 as cv
from matplotlib import pyplot as plt
import sys

N_IMAGES = 4

# read image
img = cv.imread('images\\real_images\\WSP-Ground.jpg', cv.IMREAD_GRAYSCALE)

# Otsu's thresholding
_,th1 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img,(5,5),0)
_,th2 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# adaptative thresholding with binary
th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 7, 1)

# Apply adaptive thresholding using Otsu's threshold value
_, otsu_threshold = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
th4 = cv.adaptiveThreshold(otsu_threshold, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, 0)



# plot all the images and their histograms
images = [th1, th2, th3, th4]
titles = ['Otsu thresholding',
          'Otsu thresholding with gaussian',
          'Adaptative thresholding binary',
          'Adaptative thresholding Otsu']



for i in range(N_IMAGES):
    plt.title(titles[i])
    plt.subplot(N_IMAGES, 1, i+1), plt.imshow(images[i],'gray')


plt.tight_layout()
plt.show()