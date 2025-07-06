import cv2 as cv
import numpy as np

# List of people corresponding to the labels
people = ['Samira', 'Saklain']

# Load the face detector and recognizer
haar_cascade = cv.CascadeClassifier('haar_face.xml')
face_recognizer = cv.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
face_recognizer.read('face_trained.yml')

# Open the webcam
cap = cv.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame for better performance
    frame = cv.resize(frame, (640, 480))
    
    # Convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Detect faces
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    
    # Process each detected face
    for (x,y,w,h) in faces_rect:
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Predict who this face belongs to
        label, confidence = face_recognizer.predict(face_roi)
        person_name = people[label]
        
        # Draw rectangle around the face
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        
        # Display name above the rectangle
        cv.putText(frame, person_name, (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        
        # Print confidence score in the console
        print(f'Label = {label} ({person_name}) with a confidence of {confidence:.2f}')
    
    # Display the frame
    cv.imshow('Face Recognition', frame)
    
    # Break the loop if 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
print("\nExiting Program")
cap.release()
cv.destroyAllWindows()