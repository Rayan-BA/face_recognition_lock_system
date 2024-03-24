import cv2 as cv
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets

face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

def load_data(dataset_path):
    pass

def train():
    recognizer = cv.face.LBPHFaceRecognizer.create()
    print ("\n [INFO] Training faces. It will take a few seconds. Please wait...")
    x, y = load_data("antispoof-dataset")
    recognizer.train(x, y)
    recognizer.write("./models/lbp-antispoof.yml")

def recognize():
    print(" [INFO] Recognizing faces...")

    recognizer = cv.face.LBPHFaceRecognizer.create()
    recognizer.read("./models/lbp-antispoof.yml")
    font = cv.FONT_HERSHEY_SIMPLEX
    x, y = load_data("antispoof-dataset")
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
                name = y
            else:
                name = "Unknown"
            
            cv.putText(frame, name, (x+5,y-5), font, 1, (255,255,255), 2)
            cv.putText(frame, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
        
        cv.imshow("Recognizer", frame)
        cv.setWindowProperty("Recognizer", cv.WND_PROP_TOPMOST, 1)
        # Press "ESC" for exiting video
        if cv.waitKey(1) & 0xff == 27:
            break
    video_capture.release()
    cv.destroyWindow("Recognizer")

train()