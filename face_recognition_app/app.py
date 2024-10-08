import torch_SVC
import torchEmbedder
import lbp_antispoof
import utils

# TODO:
# - Anti-spoofing

def face_recognition():
    print(" [INFO] Face Recognition")
    while True:
        print(""" [SELECT]
            1) Collect faces
            2) Create New Face Embeddings
            3) Train SVC Model
            4) Recognize Faces
            
            Press any other key to exit.""")
        op = input(" Your selection: ")
        match op:
            case "1": utils.collect_samples_onpress("./dataset", "face_rec")
            case "2": torchEmbedder.create_embeddings("./dataset", "./models/torch_embeddings.npz", skip_existing_labels=False)
            case "3": torch_SVC.train("./models/torch_embeddings.npz", "./models/torch_svc.joblib")
            case "4": torch_SVC.recognize("./models/torch_embeddings.npz", "./models/torch_svc.joblib")
            case _: break

def anti_spoof():
    while True:
        print(" [INFO] Anti-spoof")
        print(""" [SELECT]
            1) Collect faces
            2) Train Model
            3) Recognize Faces
            
            Press any other key to exit.""")
        op = input(" Your selection: ")
        match op:
            case "1": utils.collect_samples_onpress("./antispoof-dataset", "anti-spoof")
            case "2": lbp_antispoof.train()
            case "3": lbp_antispoof.recognize()
            case _: break

def main():
    print(" [INFO] Program started")
    while True:
        print(""" [SELECT]
            1) Face Recognition
            2) Anti-spoof
            
            Press any other key to exit.""")
        op = input(" Your selection: ")
        match op:
            case "1": face_recognition()
            case "2": anti_spoof()
            case _: break
    print(" [INFO] Program ended")

if __name__ == "__main__":
    main()
