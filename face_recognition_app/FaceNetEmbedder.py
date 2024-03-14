# import cv2 as cv
# from os import getenv
import numpy as np
from keras_facenet import FaceNet
import concurrent
import tensorflow as tf
from tqdm import tqdm
# from utils import loadClasses
# from dotenv import load_dotenv
tf.keras.utils.disable_interactive_logging()
from time import time
# load_dotenv()
# faces_embeddings_path = getenv("faces_embeddings_path")
# dataset_path = getenv("dataset_path")

embedder = FaceNet()

def getOneEmbedding(face_img):
    face_img = face_img.astype("float32")
    # makes array 4D, FaceNet requires this
    face_img = np.expand_dims(face_img, axis=0)
    yhat = embedder.embeddings(face_img)
    return yhat[0]

def createEmbeddings(data: tuple, save_to_path: str):
    print("[INFO] Creating new embeddings...")
    x, y = data
    
    # slowww
    # embedded_x = []
    # for img in x:
    #     embedded_x.append(getOneEmbedding(img))
    
    # t0 = time()
    # fast
    with concurrent.futures.ThreadPoolExecutor() as executor:
        embedded_x = list(tqdm(executor.map(getOneEmbedding, x)))
    # t1 = time()
    # print("tensorflow facenet: ", t1-t0)
    embedded_x = np.asarray(embedded_x)
    np.savez_compressed(save_to_path, embedded_x, y)
    print("[INFO] Embeddings done.")