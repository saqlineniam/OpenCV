import cv2 as cv


img = cv.imread('Photos/group 1.jpg')
cv.imshow('Group', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Group', gray)

blurred = cv.medianBlur(gray, 1)
cv.imshow('Blurred', blurred)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(blurred, scaleFactor=1.3, minNeighbors=2)


print(f'Number of faces found: {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Faces', img)


cv.waitKey(0)