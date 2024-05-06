from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
import cv2 as cv
import numpy as np

class mySVC:
    def __init__(self) -> None:
        self.face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.label_encoder = LabelEncoder()
        self.model = SVC(kernel="linear", probability=True)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval()
        self.finished = False

    def train(self):
        print("[INFO] Training SVC model...")
        data = np.load("./models/torch_embeddings.npz")
        embedded_x, y = data["arr_0"], data["arr_1"]
        encoder = self.label_encoder
        encoder.fit(y)
        y = encoder.transform(y)
        x_train, x_test, y_train, y_test = train_test_split(embedded_x, y, shuffle=True)
        model = self.model
        model.fit(x_train, y_train)
        with open("./models/torch_svc.joblib", "wb") as f:
            joblib.dump(model, f)
        self.finished = True
        print("[INFO] Training done.")

    def recognize(self):
        face_cascade = self.face_cascade
        model = joblib.load(open("./models/torch_svc.joblib", "rb"))
        n = np.load("./models/torch_embeddings.npz")["arr_1"]
        restnet = self.resnet
        encoder = self.label_encoder
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
                tensorImg = tensorImg.unsqueeze(0) # expands dims, equivalent to np.expand_dims(img, axis=0)
                ypred = restnet(tensorImg).detach().cpu().numpy() # .cpu() just in case gpu is used
                pred = model.predict(ypred)
                conf = int(max(model.predict_proba(ypred)[0]) * 100)
                name = encoder.inverse_transform(pred)[0]
                cv.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 1)
                cv.putText(frame, str(f"{name}  {conf}"), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3, cv.LINE_AA)
            cv.imshow("Recognizer", frame)
            cv.setWindowProperty("Recognizer", cv.WND_PROP_TOPMOST, 1)
            if cv.waitKey(1) & 0xff == 27: # ESC to exit
                break
        video_capture.release()
        cv.destroyWindow("Recognizer")

if __name__ == "__main__":
    model = mySVC()
    model.recognize()