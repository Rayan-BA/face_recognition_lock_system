import os
from shutil import rmtree
import cv2 as cv
import numpy as np
# from PIL import Image
# from time import time
# from sklearn.utils import shuffle
from uuid import uuid1
from dotenv import load_dotenv

load_dotenv()
face_cascade = cv.CascadeClassifier(os.getenv("face_cascade"))
dataset_path = os.getenv("dataset_path")

def _loadFaces(sub_dir):
    faces = []
    for img in os.listdir(sub_dir):
        path = sub_dir + "/" + img
        face = cv.imread(path)
        face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
        # FaceNet requires 160x160
        face = cv.resize(face, (160, 160))
        # print(img, face.shape)
        faces.append(face)
    return faces

def loadClasses(dir):
    x, y = [], []
    for sub_dir in os.listdir(dir):
        path = dir + "/" + sub_dir
        faces = _loadFaces(path)
        names = [sub_dir for _ in range(len(faces))]
        x.extend(faces)
        y.extend(names)
    # x, y = shuffle(x, y) can be shuffled in train_test_split() can be shuffled later
    return np.asarray(x), np.asarray(y)

def _createDir(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

def _removeDir(dir_name):
    if os.path.isdir(dir_name):
        rmtree(dir_name)

def _extractFace(frame, name, counter):
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(100,100))
    for (x,y,w,h) in faces:
        counter += 1
        face = frame[y:y+h, x:x+w]
        # face = processFace(face)
        _saveFace(face, name)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
    
    return counter

def _saveFace(face, name):
    path = f"{dataset_path}/{name}"
    # uuid is 128 bits, shift right by 102 bits to shorten
    random_id = uuid1().int >> 102
    _createDir(path)
    cv.imwrite(f"{path}/{random_id}.jpg", face)

def _drawProgBar(frame, w, h, prog):
    padding = 20
    prog_bar_start = padding
    prog_bar_end = w-padding

    cv.rectangle(frame, (prog_bar_start, h-30), (prog_bar_end, h-10), (0,0,255), 2)

    prog_bar = int(prog * prog_bar_end/100)
    # thickness -1 for fill
    if prog_bar > padding:
        cv.rectangle(frame, (prog_bar_start, h-30), (prog_bar, h-10), (0,0,255), -1)

# Open webcam and do real-time face detection and extraction
def collectFaces():
    # TODO handle existing name, currently appends to folder with same name
    # have an option to override
    name = input(" [INPUT] Enter your name: ")
    
    # if os.path.isdir(f"{dataset_path}/{name}"):
    #     ans = input(f"{name} is already saved, ")
    
    cam = cv.VideoCapture(0)
    counter = 0
    limit = 100
    frame_width  = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
    # print(frame_width, frame_height)
    while cam.isOpened():
        _, frame = cam.read()
        counter = _extractFace(frame, name, counter)
        _drawProgBar(frame, frame_width, frame_height, int((counter/limit)*100))
        cv.imshow("Webcam Capture", frame)
        cv.setWindowProperty("Webcam Capture", cv.WND_PROP_TOPMOST, 1)
        # ESC to exit
        if cv.waitKey(1) & 0xff == 27 or counter >= limit:
            break

    cam.release()
    cv.destroyWindow("Webcam Capture")