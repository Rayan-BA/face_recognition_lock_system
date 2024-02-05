import pickle
import FaceNetEmbedder
import SVC
import cv2 as cv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras_facenet import FaceNet
import tensorflow

tensorflow.keras.utils.disable_interactive_logging()

# TODO:
# - Anti-spoofing
# - Optimzie speed

def svcPred():
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    model = pickle.load(open("SVC_model.pkl", "rb"))
    faces_embeddings = np.load("faces_embeddings.npz")
    y = faces_embeddings["arr_1"]
    facenet = FaceNet()
    encoder = LabelEncoder()
    encoder.fit(y)
    encoder.transform(y)
    video_capture = cv.VideoCapture(0)
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=5, minSize=(100,100))
        for (x,y,w,h) in faces:
            img = rgb_img[y:y+h, x:x+w]
            img = cv.resize(img, (160,160))
            img = np.expand_dims(img, axis=0)
            ypred = facenet.embeddings(img)
            name_pred = model.predict(ypred)
            conf = int(max(model.predict_proba(ypred)[0]) * 100)
            name = encoder.inverse_transform(name_pred)[0]
            cv.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 1)
            cv.putText(frame, str(f"{name}  {conf}"), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3, cv.LINE_AA)
            
        cv.imshow("Recognizer", frame)
        cv.setWindowProperty("Recognizer", cv.WND_PROP_TOPMOST, 1)
        # ESC to exit
        if cv.waitKey(1) & 0xff == 27:
         break
    
    video_capture.release()
    cv.destroyWindow("Recognizer")

def main():
    while True:
        print(" [INFO] Program started")
        print(""" [SELECT]
            1) Collect faces
            2) Create New Face Embeddings
            3) Train SVC Model
            4) Recognize Faces
            
            Press any other key to exit.""")
        op = input(" Your selection: ")
        match op:
            case "1": FaceNetEmbedder.collectFaces()
            case "2": FaceNetEmbedder.createEmbedding()
            case "3": SVC.train()
            case "4": svcPred()
            case _: break
    print(" [INFO] Program ended")

main()
