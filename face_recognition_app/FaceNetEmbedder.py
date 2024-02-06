import cv2 as cv
import os
import numpy as np
from keras_facenet import FaceNet
from uuid import uuid1
from utils import createDir

faces_embeddings_path = "./models/faces_embeddings.npz"

def extractFace(frame, name, counter, limit):
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(100,100))
    for (x,y,w,h) in faces:
        counter += 1
        face = frame[y:y+h, x:x+w]
        # face = processFace(face)
        saveFace(face, name)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
    
    return counter

# def processFace(face, target_size=(160, 160)):
#     face = cv.resize(face, target_size)
#     return face

def saveFace(face, name):
    path = f"dataset/{name}"
    # uuid is 128 bits, shift right by 102 bits to shorten
    random_id = uuid1().int >> 102
    createDir(path)
    cv.imwrite(f"{path}/{random_id}.jpg", face)

def loadFaces(dir):
    faces = []
    for img in os.listdir(dir):
        path = dir + "/" + img
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
        faces = loadFaces(path)
        names = [sub_dir for _ in range(len(faces))]
        x.extend(faces)
        y.extend(names)
    return np.asarray(x), np.asarray(y)

def drawProgBar(frame, w, h, prog):
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
    name = input("Enter your name: ")
    cam = cv.VideoCapture(0)
    counter = 0
    limit = 100
    frame_width  = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(frame_width, frame_height)
    while cam.isOpened():
        _, frame = cam.read()
        counter = extractFace(frame, name, counter, limit)
        drawProgBar(frame, frame_width, frame_height, int((counter/limit)*100))
        cv.imshow("Webcam Capture", frame)
        cv.setWindowProperty("Webcam Capture", cv.WND_PROP_TOPMOST, 1)
        # ESC to exit
        if cv.waitKey(1) & 0xff == 27 or counter >= limit:
            break

    cam.release()
    cv.destroyWindow("Webcam Capture")

def getEmbedding(embedder, face_img):
    face_img = face_img.astype("float32")
    # makes array 4D, FaceNet requires this
    face_img = np.expand_dims(face_img, axis=0)
    yhat = embedder.embeddings(face_img)
    return yhat[0]

def createEmbedding():
    print("[INFO] Creating new embeddings...")
    embedder = FaceNet()
    x, y = loadClasses("dataset")
    embedded_x = []
    for img in x:
        embedded_x.append(getEmbedding(embedder, img))
    embedded_x = np.asarray(embedded_x)
    np.savez_compressed(faces_embeddings_path, embedded_x, y)
    print("[INFO] Embeddings done.")
