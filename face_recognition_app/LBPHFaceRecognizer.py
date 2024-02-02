import cv2 as cv
import numpy as np
from utils import getFacesIdsNames

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
dataset_path = "dataset"

def train():
    recognizer = cv.face.LBPHFaceRecognizer.create()
    print ("\n [INFO] Training faces. It will take a few seconds. Please wait...")
    faces,ids,_ = getFacesIdsNames(dataset_path)
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer.yml')
    print(f"\n [INFO] {len(np.unique(ids))} faces trained.")

def recognize():
    print(" [INFO] Recognizing faces...")

    recognizer = cv.face.LBPHFaceRecognizer.create()
    recognizer.read('trainer.yml')
    font = cv.FONT_HERSHEY_SIMPLEX
    _, _, names = getFacesIdsNames(dataset_path)
    video_capture = cv.VideoCapture(0)
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)
        
        faces = face_cascade.detectMultiScale(frame_gray, minSize = (100, 100))
        for(x,y,w,h) in faces:
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            id, doubt = recognizer.predict(frame_gray[y:y+h,x:x+w])
            
            confidence = round(100 - doubt)
            if confidence > 50:
                name = names[id]
            else:
                name = "Unknown"
            
            cv.putText(frame, name, (x+5,y-5), font, 1, (255,255,255), 2)
            cv.putText(frame, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
        
        cv.imshow('Recognizer', frame)
        cv.setWindowProperty('Recognizer', cv.WND_PROP_TOPMOST, 1)
        # Press 'ESC' for exiting video
        if cv.waitKey(1) & 0xff == 27:
            break
    video_capture.release()
    cv.destroyWindow("Recognizer")
