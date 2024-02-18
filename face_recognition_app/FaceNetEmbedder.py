# import cv2 as cv
from os import getenv
import numpy as np
from keras_facenet import FaceNet
import concurrent
from utils import loadClasses
from dotenv import load_dotenv

load_dotenv()
faces_embeddings_path = getenv("faces_embeddings_path")
dataset_path = getenv("dataset_path")

embedder = FaceNet()
def getOneEmbedding(face_img):
    face_img = face_img.astype("float32")
    # makes array 4D, FaceNet requires this
    face_img = np.expand_dims(face_img, axis=0)
    yhat = embedder.embeddings(face_img)
    return yhat[0]

def createEmbeddings():
    print("[INFO] Creating new embeddings...")
    x, y = loadClasses(dataset_path)
    
    # slowww
    # embedded_x = []
    # for img in x:
    #     embedded_x.append(getOneEmbedding(img))
    
    # fast
    with concurrent.futures.ThreadPoolExecutor() as executor:
        embedded_x = list(executor.map(getOneEmbedding, x))

    embedded_x = np.asarray(embedded_x)
    np.savez_compressed(faces_embeddings_path, embedded_x, y)
    print("[INFO] Embeddings done.")
