import cv2 as cv
import numpy as np
from os import listdir
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
import joblib

class LBPHSpoofDetector:
    def __init__(self, radius, points) -> None:
        self.model = SVC(kernel="linear", probability=True)
        self.face_cascade = cv.CascadeClassifier("./haarcascade_frontalface_default.xml")
        self.radius = radius
        self.points = points

    def train(self):
        print ("\n [INFO] Training faces. It will take a few seconds. Please wait...")
        model = self.model
        images, labels = self.load_data("./antispoof-dataset")

        images, labels = shuffle(images, labels, random_state=42)
        
        data, encoded_labels = [], []
        for lbl in labels:
            if lbl == "real": encoded_labels.append(0)
            elif lbl == "spoof": encoded_labels.append(1)
        
        for img in images:
            data.append(self.extract_lbp_features(img, self.radius, self.points))
        
        model.fit(data, labels)
        with open("./models/svc_antispoof.joblib", "wb") as f:
            joblib.dump(model, f)
    
    def extract_lbp_features(self, image, radius=1, num_points=8, eps=1e-7):
        lbp = local_binary_pattern(image, num_points, radius, method="uniform")

        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))

        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist

    def _load_images(self, sub_dir):
        faces = []
        for img in listdir(sub_dir):
            path = sub_dir + "/" + img
            face = cv.imread(path, 0)
            face = cv.resize(face, (160, 160))
            faces.append(face)
        return faces

    def load_data(self, dir):
        print(" [INFO] Loading classes...")
        x, y = [], []
        for sub_dir in listdir(dir):
            path = dir + "/" + sub_dir
            faces = self._load_images(path)
            labels = [sub_dir for _ in range(len(faces))]
            x.extend(faces)
            y.extend(labels)
        print(" [INFO] Loading done.")
        return np.asarray(x), np.asarray(y)

    def recognize(self):
        print(" [INFO] Recognizing faces...")
        model = joblib.load(open("./models/svc_antispoof.joblib", "rb"))
        video_capture = cv.VideoCapture(0)
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(frame_gray, minSize = (100, 100))
            for(x,y,w,h) in faces:
                lbp = self.extract_lbp_features(cv.resize(frame_gray[y:y+h,x:x+w], (160, 160)), self.radius, self.points)
                pred = model.predict(lbp.reshape(1, -1))
                conf = int(max(model.predict_proba(lbp.reshape(1, -1))[0]) * 100)
                # print(f"{'real' if pred == 0 else 'spoof'}, with conf: {conf}")
                cv.putText(frame, f"{'real' if pred == 0 else 'spoof'} conf: {conf}", (x+5,y-5), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv.imshow("Recognizer", frame)
            cv.setWindowProperty("Recognizer", cv.WND_PROP_TOPMOST, 1)
            # Press "ESC" for exiting video
            if cv.waitKey(1) & 0xff == 27:
                break
        video_capture.release()
        # cv.destroyWindow("Recognizer")

    def eval(self):
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        from tqdm import tqdm

        x, y = self.load_data("./antispoof-dataset")

        model = self.model
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=42, test_size=0.2)
        X_train, X_test, y_train, y_test = np.asarray(X_train), np.asarray(X_test), np.asarray(y_train), np.asarray(y_test)

        # print(X_train.shape, y_train.shape)
        # print(X_test.shape, y_test.shape)
        # exit()
        
        radius_pool = np.arange(1, 9)
        points_pool = np.arange(3, 25)

        encoded_labels = []
        for lbl in y_train:
            if lbl == "real": encoded_labels.append(0)
            elif lbl == "spoof": encoded_labels.append(1) 
        
        ey_test = []
        for lbl_test in y_test:
            if lbl_test == "real": ey_test.append(0)
            elif lbl_test == "spoof": ey_test.append(1)
        y_test = np.asarray(ey_test)

        accuracy, cm, precision, recall, f1 = [],[],[],[],[]
        
        for radius in tqdm(radius_pool):
            for points in tqdm(points_pool):

                data = []
                for train_img in tqdm(X_train):
                    data.append(self.extract_lbp_features(train_img, radius, points))
                
                model.fit(data, encoded_labels)

                y_preds = []
                for img_test in tqdm(X_test):
                    # Predict the labels of the test set
                    y_pred = model.predict(self.extract_lbp_features(img_test, radius, points).reshape(1, -1))
                    y_preds.append(y_pred)
                y_preds = np.asarray(y_preds).reshape(559,)
                
                print(y_preds, y_preds.shape, y_test, y_test.shape)

                # Compute the accuracy of the model
                accuracy.append((np.mean(y_preds == y_test)))

                # Compute the confusion matrix
                cm.append(confusion_matrix(y_test, y_preds))

                # # Compute the precision, recall, and F1 score
                precision.append(precision_score(y_test, y_preds, average='binary'))
                recall.append(recall_score(y_test, y_preds, average='binary'))
                f1.append(f1_score(y_test, y_preds, average='binary'))

        best_params = (radius_pool[np.max(f1)], points_pool[np.max(f1)])
        # Print the evaluation metrics
        print(f"Best params: radius={best_params[0]}, points={best_params[1]}")
        print(f"Accuracy: max={np.max(accuracy) * 100} mean={np.mean(accuracy)} std={np.std(accuracy)}")
        print(f"Confusion Matrix: max={np.max(cm) * 100} mean={np.mean(cm)} std={np.std(cm)}")
        print(f"Precision: max={np.max(precision) * 100} mean={np.mean(precision)} std={np.std(precision)}")
        print(f"Recall: max={np.max(recall) * 100} mean={np.mean(recall)} std={np.std(recall)}")
        print(f"F1 Score: max={np.max(f1) * 100} mean={np.mean(f1)} std={np.std(f1)}")

if __name__ == "__main__":
    LBPHSpoofDetector(8, 24).eval()