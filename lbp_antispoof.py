import cv2 as cv
import numpy as np
from os import listdir
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
import joblib

class LBPHSpoofDetector:
    def __init__(self, C, radius, points) -> None:
        self.face_cascade = cv.CascadeClassifier("./haarcascade_frontalface_default.xml")
        self.radius = radius
        self.points = points
        self.C = C
        self.model = SVC(C=self.C, kernel="rbf", probability=True)

    def train(self, model_file):
        print ("\n [INFO] Training faces. It will take a few seconds. Please wait...")
        model = self.model
        images, labels = self.load_data("./antispoof-dataset")

        X_train, X_test, y_train, y_test = train_test_split(images, labels, shuffle=True, random_state=42, test_size=0.05)
        
        data, encoded_labels = [], []
        for lbl in y_train:
            if lbl == "real": encoded_labels.append(0)
            elif lbl == "spoof": encoded_labels.append(1)
        
        for img in X_train:
            data.append(self.extract_lbp_features(img, self.radius, self.points))

        model.fit(data, encoded_labels)
        with open(model_file, "wb") as f:
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

    def recognize(self, model_file):
        print(" [INFO] Recognizing faces...")
        model = joblib.load(open(model_file, "rb"))
        video_capture = cv.VideoCapture(0)
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(frame_gray, minSize = (100, 100))
            for(x,y,w,h) in faces:
                lbp = self.extract_lbp_features(cv.resize(frame_gray[y:y+h,x:x+w], (160, 160)), self.radius, self.points)
                pred = model.predict(lbp.reshape(1, -1))
                conf = int(max(model.predict_proba(lbp.reshape(1, -1))[0]*100))
                # print(pred, conf)
                # print(f"{'real' if pred == 0 else 'spoof'}, with conf: {conf}")
                cv.putText(frame, f"{model_file[23:29]} {'real' if pred == 0 else 'spoof'} conf: {conf}", (x+5,y-5), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv.imshow("Recognizer", frame)
            cv.setWindowProperty("Recognizer", cv.WND_PROP_TOPMOST, 1)
            # Press "ESC" for exiting video
            if cv.waitKey(1) & 0xff == 27:
                break
        video_capture.release()
        # cv.destroyWindow("Recognizer")

    def eval(self):
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
        from sklearn.model_selection import cross_val_score
        from tqdm import tqdm

        x, y = self.load_data("./antispoof-dataset")

        X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=42, test_size=0.2)
        X_train, X_test, y_train, y_test = np.asarray(X_train), np.asarray(X_test), np.asarray(y_train), np.asarray(y_test)

        # print(X_train.shape, y_train.shape)
        # print(X_test.shape, y_test.shape)
        # exit()
        
        radius_pool = np.arange(4, 9)
        points_pool = np.arange(7, 25)
        C_pool = [10, 100, 1000]

        encoded_labels = []
        for lbl in y_train:
            if lbl == "real": encoded_labels.append(0)
            elif lbl == "spoof": encoded_labels.append(1)
        
        ey_test = []
        for lbl_test in y_test:
            if lbl_test == "real": ey_test.append(0)
            elif lbl_test == "spoof": ey_test.append(1)
        y_test = np.asarray(ey_test)

        accuracy, cm, precision, recall, f1, cross_val, roc_auc  = [],[],[],[],[],[],[]
        
        for C_factor in tqdm(C_pool):
            model = SVC(C=C_factor, kernel="rbf", probability=True)
            for radius in tqdm(radius_pool):
                for points in tqdm(points_pool):
                    print(C_factor, radius, points)

                    data = []
                    for train_img in X_train:
                        data.append(self.extract_lbp_features(train_img, radius, points))
                    
                    cross_val.append(cross_val_score(model, data, encoded_labels, cv=5))

                    model.fit(data, encoded_labels)

                    y_preds = []
                    y_scores = []
                    for img_test in X_test:
                        # Predict the labels of the test set
                        y_pred = model.predict(self.extract_lbp_features(img_test, radius, points).reshape(1, -1))
                        y_preds.append(y_pred)
                        y_score = model.predict_proba(self.extract_lbp_features(img_test, radius, points).reshape(1, -1))[:,1]
                        y_scores.append(y_score)
                    y_preds = np.asarray(y_preds)
                    
                    # Compute the accuracy of the model
                    accuracy.append((np.mean(y_preds == y_test)))

                    # Compute the confusion matrix
                    cm.append(confusion_matrix(y_test, y_preds))

                    # # Compute the precision, recall, and F1 score
                    precision.append(precision_score(y_test, y_preds, average='binary'))
                    recall.append(recall_score(y_test, y_preds, average='binary'))
                    f1.append(f1_score(y_test, y_preds, average='binary'))

                    # Compute the ROC AUC score
                    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
                    roc_auc.append(auc(fpr, tpr))

            # best_params = (C_pool[np.argmax(f1)], radius_pool[np.argmax(f1)], points_pool[np.argmax(f1)])
            # Print the evaluation metrics
            # print(f"Best params: C={best_params[0]}, radius={best_params[1]}, points={best_params[2]}")
        scores = f"""
Accuracy: max={np.max(accuracy) * 100} at {np.argmax(accuracy)}, mean={np.mean(accuracy) * 100}, std={np.std(accuracy) * 100}
Cross validation: max={np.max(cross_val)} at {np.argmax(cross_val)}, mean={np.mean(cross_val)}, std={np.std(cross_val)}
Confusion Matrix: max={np.max(cm)} at {np.argmax(cm)}, mean={np.mean(cm)}, std={np.std(cm)}
Precision: max={np.max(precision)} at {np.argmax(precision)}, mean={np.mean(precision)}, std={np.std(precision)}
Recall: max={np.max(recall)} at {np.argmax(recall)}, mean={np.mean(recall)}, std={np.std(recall)}
F1 Score: max={np.max(f1)} at {np.argmax(f1)}, mean={np.mean(f1)}, std={np.std(f1)}
ROC AUC: max={np.max(roc_auc)} at {np.argmax(roc_auc)}, mean={np.mean(roc_auc)}, std={np.std(roc_auc)}"""

        with open("scores.txt", "a") as file:
            file.write(scores)

if __name__ == "__main__":
    '''
    make sure parameters match antispoof model
    acc     C=1000, R=7, P=10
    prec    C=1000, R=6, P=11
    rec	    C=100,  R=8, P=12
    f1 	    C=1000, R=7, P=10
    roc auc	C=100,  R=8, P=16
    
    acc 50  C=100,  R=4, P=14

    f1_acc and roc_auc is best so far
    '''
    LBPHSpoofDetector(100, 4, 14).recognize("./models/svc_antispoof_acc50.joblib")
    # models = [
    #         {"file":"svc_antispoof_f1_acc_v2.joblib", "C": 1000, "R":7, "P":10},
    #         {"file":"svc_antispoof_prec_v2.joblib", "C": 1000, "R":6, "P":11},
    #         {"file":"svc_antispoof_rec_v2.joblib", "C": 100, "R":8, "P":12},
    #         {"file":"svc_antispoof_roc_auc_v2.joblib", "C": 100, "R":8, "P":16},
    #     ]
    
    # for model in models:
    #     LBPHSpoofDetector(model["C"], model["R"], model["P"]).recognize(f"./models/{model['file']}")