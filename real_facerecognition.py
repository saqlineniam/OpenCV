import cv2 as cv
import numpy as np

# List of people corresponding to the labels
people = ['Samira', 'Saklain']
DIR = r'E:\Computer Vision\My photos'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
face_recognizer.read('face_trained.yml')

img = cv.imread(r"C:\Users\saqli\Downloads\PXL_20240329_201154158.jpg")
if img is None:
    print("Error: Could not load image. Please check the file path.")
    exit(1)

# Resize the image to a smaller size for better display
img = cv.resize(img, (640, 480))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect faces in the image
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

# For each face found
for (x,y,w,h) in faces_rect:
    # Extract the face ROI
    face_roi = gray[y:y+h, x:x+w]
    
    # Predict who this face belongs to
    label, confidence = face_recognizer.predict(face_roi)
    person_name = people[label]
    
    # Print the prediction results
    print(f'Label = {label} ({person_name}) with a confidence of {confidence:.2f}')
    
    # Draw rectangle around the face
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    
    # Put the name above the rectangle
    cv.putText(img, person_name, (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

cv.imshow('Detected Face', img)


cv.waitKey(0)