import random 
import cv2 
import matplotlib.pyplot as plt

# salt-and-pepper noise can 
# be applied only to grayscale images 
# Reading the color image in grayscale image 
img = cv2.imread('data\\temp.png') 
  
img = add_noise(img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite('data\\temp.png', img)
plt.imshow(img)
plt.show()