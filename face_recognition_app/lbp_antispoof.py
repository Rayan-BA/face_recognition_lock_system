import cv2 as cv
from utils import load_data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class LBPHSpoofDetector:
    def __init__(self, ) -> None:
        self.recognizer = cv.face.LBPHFaceRecognizer.create(radius=1, neighbors=8, grid_x=2, grid_y=2, threshold=70)
        self.label_encoder = LabelEncoder()
        self.face_cascade = cv.CascadeClassifier("./haarcascade_frontalface_default.xml")

    def train(self):
        print ("\n [INFO] Training faces. It will take a few seconds. Please wait...")
        recognizer = self.recognizer
        x, y = load_data("./antispoof-dataset")
        encoder = self.label_encoder
        encoder.fit(y)
        y = encoder.transform(y)
        X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=42)
        recognizer.train(X_train, y_train)
        recognizer.write("./models/lbp-antispoof.yml")

    def recognize(self):
        print(" [INFO] Recognizing faces...")
        recognizer = self.recognizer
        recognizer.read("./models/lbp-antispoof.yml")
        video_capture = cv.VideoCapture(0)
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            frame_gray = cv.equalizeHist(frame_gray)
            faces = self.face_cascade.detectMultiScale(frame_gray, minSize = (100, 100))
            for(x,y,w,h) in faces:
                cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                # conf is actually distance, so lower is better
                pred, conf = recognizer.predict(frame_gray[y:y+h,x:x+w])
                cv.putText(frame, f"{'real' if pred == 0 else 'spoof'}", (x+5,y-5), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv.imshow("Recognizer", frame)
            cv.setWindowProperty("Recognizer", cv.WND_PROP_TOPMOST, 1)
            # Press "ESC" for exiting video
            if cv.waitKey(1) & 0xff == 27:
                break
        video_capture.release()
        cv.destroyWindow("Recognizer")

def evaluate_model():
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import numpy as np
    from tqdm import tqdm

    x, y = load_data("./antispoof-dataset")
    
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=42)

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    
    threshold = 70
    radius = 1
    neighbors = 8
    grid_ys = np.arange(2, 9, 2)
    grid_xs = np.arange(2, 9, 2)

    for grid_y in tqdm(grid_ys):
        for grid_x in tqdm(grid_xs):
            recognizer = cv.face.LBPHFaceRecognizer.create(threshold=threshold,radius=radius, neighbors=neighbors, grid_y=grid_y, grid_x=grid_x)
            recognizer.train(X_train, y_train)

            y_pred = []

            for img in X_test:
                # Make predictions on the test set
                y_pred.append(recognizer.predict(img)[0])

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_list.append(accuracy)

            # Calculate precision
            precision = precision_score(y_test, y_pred, average="macro")
            precision_list.append(precision)

            # Calculate recall
            recall = recall_score(y_test, y_pred, average="macro")
            recall_list.append(recall)

            # Calculate F1 score
            f1 = f1_score(y_test, y_pred, average="macro")
            f1_list.append(f1)

    # best_threshold = thresholds[np.argmax(f1_list)]
    best_params = (grid_ys[np.argmax(f1_list)], grid_xs[np.argmax(f1_list)])

    # Calculate the average performance metric for each threshold value
    avg_accuracy = np.mean(accuracy_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)

    # Print the evaluation metrics
    print(f"Best parameters: radius={radius}, neighbors={neighbors}, grid_y={best_params[0]}, grid_x={best_params[1]}")
    print(f"Accuracy: {np.max(accuracy_list):.2f} ({avg_accuracy:.2f} ± {np.std(accuracy_list):.2f})")
    print(f"Precision: {np.max(precision_list):.2f} ({avg_precision:.2f} ± {np.std(precision_list):.2f})")
    print(f"Recall: {np.max(recall_list):.2f} ({avg_recall:.2f} ± {np.std(recall_list):.2f})")
    print(f"F1 Score: {np.max(f1_list):.2f} ({avg_f1:.2f} ± {np.std(f1_list):.2f})")
