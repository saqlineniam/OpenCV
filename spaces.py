import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('Photos/park.jpg')
cv.imshow('park', img)

#BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)

#BGR to Gray
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#BGR to LAB
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)

#BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB', rgb)

plt.imshow(rgb)
plt.show()


#GRAY to BGR
#gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
#cv.imshow('Gray_BGR', gray_bgr)
 


cv.waitKey(0)