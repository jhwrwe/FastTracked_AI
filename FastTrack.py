import cv2
import os
import numpy as np


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

dataset_folder = "faces_dataset"


def load_dataset(folder):
    faces = []
    labels = []
    label_mapping = {}

    for label, person_name in enumerate(os.listdir(folder)):
        person_folder = os.path.join(folder, person_name)
        label_mapping[label] = person_name
        
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            labels.append(label)
    
    return faces, labels, label_mapping


faces, labels, label_mapping = load_dataset(dataset_folder)


face_recognizer.train(faces, np.array(labels))


video_capture = cv2.VideoCapture(0)


students_present = []

while True:

    ret, frame = video_capture.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y + h, x:x + w]


        label, confidence = face_recognizer.predict(face_roi)

        if confidence < 100:  
            name = label_mapping[label]
            if name not in students_present:
                students_present.append(name)


            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imshow('Face Recognition', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


print("Students present:", students_present)


video_capture.release()
cv2.destroyAllWindows()
