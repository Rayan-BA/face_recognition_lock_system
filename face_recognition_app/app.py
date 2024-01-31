import cv2 as cv
import os
from time import time
import numpy as np
from PIL import Image

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

def detectFace():
    print(" [INFO] Detecting faces...")
    video_capture = cv.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)
        # Detect faces
        faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(100,100))
        for (x,y,w,h) in faces:
            frame = cv.rectangle(frame, (x, y), (x+w, y+h), (50,50,255), 1)
        cv.imshow("Detector", frame)
        cv.setWindowProperty('Detector', cv.WND_PROP_TOPMOST, 1)
        # ESC to exit
        if cv.waitKey(1) & 0xff == 27:
         break
    
    video_capture.release()
    cv.destroyWindow("Detector")

def readIdFromFile():
    file_path = "last_id.txt"
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with open(file_path, "w") as file:
            file.write("1")
    
    with open(file_path, "r") as file:
        return file.read()

def incrementId(id):
    id += 1
    with open("last_id.txt", "w") as file:
        file.write(str(id))

def createDir(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

def collectFaceSamples():
    print(" [INFO] Collecting face samples...")

    dir_name = "datasets"
    createDir(dir_name)

    id = readIdFromFile()

    name = str(input("\n [INPUT] Enter your name: "))

    # Flip video image vertically
    # frame = cv.flip(frame, 0)
    
    time_limit = 10

    t0 = time()
    video_capture = cv.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)
        faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(100,100))

        t1 = time()
        seconds = t1 - t0
        for (x,y,w,h) in faces:
            cv.putText(frame, f"{str(int(seconds))}/{time_limit}", (30, 40), cv.FONT_HERSHEY_PLAIN, 2, (50,50,255), 2)
            cv.rectangle(frame, (x,y), (x+w,y+h), (50,50,255), 1)
            # Save the captured image into the data folder
            cv.imwrite(f"{dir_name}/{name}.{id}.{seconds}.jpg", frame_gray[y:y+h,x:x+w])
        cv.imshow("Collector", frame)
        cv.setWindowProperty('Collector', cv.WND_PROP_TOPMOST, 1)
        # Take 4 face sample and stop video
        if seconds > time_limit or cv.waitKey(1) & 0xff == 27:
            break

    incrementId(int(id))

    video_capture.release()
    cv.destroyWindow("Collector")

def getFacesIdsNames(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    names = []
    for imagePath in imagePaths:
        # grayscale
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        name = os.path.split(imagePath)[-1].split(".")[0]
        faces = face_cascade.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
            names.append(name)
    return faceSamples,ids, names

def trainRecognizer():
    recognizer = cv.face.LBPHFaceRecognizer.create(radius=1,grid_x=8,grid_y=8,neighbors=8,threshold=4096)
    print ("\n [INFO] Training faces. It will take a few seconds. Please wait...")
    faces,ids,_ = getFacesIdsNames("datasets")
    recognizer.train(faces, np.array(ids))# Save the model into trainer/trainer.yml
    recognizer.write('trainer.yml') # Print the numer of faces trained and end program
    print(f"\n [INFO] {len(np.unique(ids))} faces trained.")

def recognizeFace():
    print(" [INFO] Recognizing faces...")

    recognizer = cv.face.LBPHFaceRecognizer.create()
    recognizer.read('trainer.yml')
    font = cv.FONT_HERSHEY_SIMPLEX
    id = 0
    _, _, names = getFacesIdsNames("datasets")
    video_capture = cv.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)
        
        faces = face_cascade.detectMultiScale(frame_gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (100, 100))
        for(x,y,w,h) in faces:
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            id, doubt = recognizer.predict(frame_gray[y:y+h,x:x+w])
            
            confidence = round(100 - doubt)
            if confidence > 50:
                name = names[id]
            else:
                name = "Unknown"
            
            cv.putText(
                        frame, 
                        name, 
                        (x+5,y-5), 
                        font, 
                        1, 
                        (255,255,255), 
                        2
                    )
            cv.putText(
                        frame, 
                        str(confidence), 
                        (x+5,y+h-5), 
                        font, 
                        1, 
                        (255,255,0), 
                        1
                    )  
        
        cv.imshow('Recognizer', frame)
        cv.setWindowProperty('Recognizer', cv.WND_PROP_TOPMOST, 1)
        # Press 'ESC' for exiting video
        if cv.waitKey(1) & 0xff == 27:
            break
    video_capture.release()
    cv.destroyWindow("Recognizer")

def main():
    print(" [INFO] Program started")
    print(""" [SELECT]
            1) Detect Faces
            2) Collect Face Samples
            3) Train Recognizer
            4) Recognize Faces
            
            Press any other key to exit.""")
    while True:
        op = int(input("Your selection: "))
        match op:
            case 1: detectFace()
            case 2: collectFaceSamples()
            case 3: trainRecognizer()
            case 4: recognizeFace()
            case _: break
    cv.destroyAllWindows()
    print(" [INFO] Program ended")

main()
