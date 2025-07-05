import cv2 as cv

img = cv.imread('Photos/cat.jpg')
cv.imshow('Cat', img)

#converting image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#blur
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

#edge cascade
canny = cv.Canny(blur, 125,175)
cv.imshow('Canny', canny)

#dilating the image
dilated = cv.dilate(canny, (3,3), iterations = 10)
cv.imshow('dilated', dilated)

#erode  
eroded = cv.erode(dilated, (3,3),iterations = 10)
cv.imshow('erode', eroded)

#resize
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

#Cropping
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)




cv.waitKey(0)
