from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np
import joblib
from os import getenv
from dotenv import load_dotenv
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
import cv2 as cv
load_dotenv()

faces_embeddings_path = getenv("faces_embeddings_path")
SVC_model_path = getenv("SVC_model_path")

def train():
    print("[INFO] Training SVC model...")
    data = np.load(faces_embeddings_path)
    embedded_x, y = data["arr_0"], data["arr_1"]

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)

    x_train, x_test, y_train, y_test = train_test_split(embedded_x, y, shuffle=True)

    model = SVC(kernel="linear", probability=True)
    model.fit(x_train, y_train)
    
    with open(SVC_model_path, "wb") as f:
        joblib.dump(model, f)
    print("[INFO] Training done.")

def recognize():
    face_cascade = cv.CascadeClassifier(getenv("face_cascade"))
    model = joblib.load(open(SVC_model_path, "rb"))
    embeddings = np.load(faces_embeddings_path)
    n = embeddings["arr_1"]
    restnet = InceptionResnetV1(pretrained="vggface2").eval()
    encoder = LabelEncoder()
    encoder.fit(n)
    encoder.transform(n)
    video_capture = cv.VideoCapture(0)
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        faces = face_cascade.detectMultiScale(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), scaleFactor=1.3, minNeighbors=5, minSize=(100,100))
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        for (x, y, w, h) in faces:
            img = rgb_img[y:y+h, x:x+w]
            img = cv.resize(img, (160,160))
            tensorImg = transforms.ToTensor()(img)
            # expands dims, equivalent to np.expand_dims(img, axis=0)
            tensorImg = tensorImg.unsqueeze(0)
            # .cpu() just in case gpu is used
            ypred = restnet(tensorImg).detach().cpu().numpy()
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
