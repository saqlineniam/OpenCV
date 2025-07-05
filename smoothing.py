import cv2 as cv
import numpy as np

img = cv.imread('Photos/cats.jpg')
cv.imshow('cats', img)

#Averaging
average = cv.blur(img, (3,3))
cv.imshow('Average', average)


#Gaussian blur

gauss = cv.GaussianBlur(img, (3,3), 0)
cv.imshow('Gaussian', gauss)


#median blur
median = cv.medianBlur(img, 3)
cv.imshow('Median', median)

#bilateral blur
bilateral = cv.bilateralFilter(img, 10,70, 20)
cv.imshow('Bilateral', bilateral)


cv.waitKey(0)