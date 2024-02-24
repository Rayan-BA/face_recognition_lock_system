import torch_SVC
import torchEmbedder
import utils
from os import getenv
from dotenv import load_dotenv

# TODO:
# - Anti-spoofing

load_dotenv()
embeddings_path = getenv("faces_embeddings_path")
SVC_model_path = getenv("SVC_model_path")
dataset_path = getenv("dataset_path")

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
            case "2": torchEmbedder.createEmbeddings(dataset_path, embeddings_path)
            case "3": torch_SVC.train()
            case "4": torch_SVC.recognize()
            case _: break
    print(" [INFO] Program ended")

if __name__ == "__main__":
    main()
