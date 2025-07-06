import cv2 as cv
import numpy as np

# List of people corresponding to the labels
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

haar_cascade = cv.CascadeClassifier('haar_face.xml')

# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'E:\Computer Vision\Faces\val\madonna\1.jpg')
if img is None:
    print("Error: Could not load image. Please check the file path.")
    exit(1)

# Resize the image to a smaller size for better display
img = cv.resize(img, (800, 600))
cv.imshow('Person', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#detect faces
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]
    label, confidence = face_recognizer.predict(faces_roi)
    person_name = people[label]
    print(f'Label = {label} ({person_name}) with a confidence of {confidence}')
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    cv.putText(img, person_name, (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)

cv.imshow('Detected Face', img)


cv.waitKey(0)