import SVC
import tensorflow as tf
import FaceNetEmbedder
import utils
from os import getenv
from dotenv import load_dotenv
from os import getenv

tf.keras.utils.disable_interactive_logging()

# TODO:
# - Anti-spoofing
# - Optimzie speed (multithreading or multiprocessing) done but maybe improve?
# - Lower keras load time?


load_dotenv()
faces_embeddings_path = getenv("faces_embeddings_path")
SVC_model_path = getenv("SVC_model_path")


def main():
    print(" [INFO] Program started")
    while True:
        print(""" [SELECT]
            1) Collect faces
            2) Create New Face Embeddings
            3) Train SVC Model
            4) Recognize Faces
            
            Press any other key to exit.""")
        op = input(" Your selection: ")
        match op:
            case "1": utils.collectFaces()
            case "2": FaceNetEmbedder.createEmbeddings()
            case "3": SVC.train()
            case "4": SVC.recognize()
            case _: break
    print(" [INFO] Program ended")

if __name__ == "__main__":
    main()
