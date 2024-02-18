import os
import cv2 as cv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from skimage.feature import local_binary_pattern
from dotenv import load_dotenv
import numpy as np
import joblib
from tqdm import tqdm
from utils import extractFace

load_dotenv()

"""
perplexity.ai: mixtral-7b

To improve speed, you could consider using transfer learning and fine-tuning a pre-trained CNN model on your own dataset.
Transfer learning allows you to leverage knowledge learned from large datasets and apply it to smaller, task-specific datasets.
Fine-tuning involves adjusting the weights of the last few layers of the pre-trained model to better fit the characteristics of your dataset.
By doing so, you can significantly reduce the amount of time required for training while still achieving high accuracy.

"""

# Define paths
data_dir = "D:/CelebA_Spoof"

def extract_lbp_features(image_directory):
    # Extracts Local Binary Pattern (LBP) features from all images within the specified directory
    features = []
    labels = []
    for subdir in os.listdir(image_directory):
        path = os.path.join(image_directory, subdir)
        for filename in tqdm(os.listdir(path)):
            filepath = os.path.join(path, filename)
            img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
            if img is None: continue
            img = extractFace(img)
            if img is None: continue
            # print(img, type(img), filename)
            img = cv.resize(img, (128, 128))
            lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
            # hist, _ = np.histogram(lbp.ravel(), bins=[0]+range(91)+[2**8-1], density=True)
            hist, _ = np.histogram(lbp, bins=[0, 8, 16, 32, 64, 128], range=(0, 129), density=True)
            features.append(hist)
            labels.append(subdir == "live")
    np.savez_compressed("./models/anti-spoof-lbp-features.npz", np.asarray(features), np.asarray(labels).astype(int))

def extract_lbp_features_webcam(frame):
    lbp = local_binary_pattern(frame, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=[0, 8, 16, 32, 64, 128], range=(0, 129), density=True)
    return hist

def create_training_testing_dirs(data_dir):
    # Create directories if they don't exist
    train_images_dir = os.path.join(data_dir, "training")
    test_images_dir = os.path.join(data_dir, "testing")
    if not os.path.exists(train_images_dir):
        os.makedirs(train_images_dir)
    if not os.path.exists(test_images_dir):
        os.makedirs(test_images_dir)

# TODO fix
def split_training_testing(data_dir):
    # Divide live and spoof images randomly into train and test sets
    live_dir = os.path.join(data_dir, "live")
    spoof_dir = os.path.join(data_dir, "spoof")
    train_images_dir = os.path.join(data_dir, "training")
    test_images_dir = os.path.join(data_dir, "testing")
    for dir_name, split_ratio in zip((live_dir, spoof_dir), (0.75, 0.75)):
        files = os.listdir(dir_name)
        num_files = len(files)
        train_indices = list(np.random.choice(num_files, size=int(num_files * split_ratio), replace=False))
        
        # Move selected files to training directory
        for index in train_indices:
            file_name = files[index]
            src_file = os.path.join(dir_name, file_name)
            dst_folder = os.path.join(train_images_dir, dir_name.split("/")[-1])
            dst_file = os.path.join(dst_folder, file_name)
            
            os.makedirs(dst_folder, exist_ok=True)
            os.rename(src_file, dst_file)

        # Move remaining files to testing directory
        for i, file_name in enumerate(files):
            if i not in train_indices:
                src_file = os.path.join(dir_name, file_name)
                dst_folder = os.path.join(test_images_dir, dir_name.split("/")[-1])
                dst_file = os.path.join(dst_folder, file_name)

                os.makedirs(dst_folder, exist_ok=True)
                os.rename(src_file, dst_file)

def train_SVC():
    data = np.load("./models/anti-spoof-lbp-features.npz")
    X, y = data["arr_0"], data["arr_1"]

    # Split into train & validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize SVC classifier
    model = SVC(kernel="linear", probability=True)

    # Fit the classifier
    model.fit(X_train, y_train)
    
    # Evaluate performance on validation set
    # y_pred = model.predict(X_val)
    # print("Validation Report:\n", classification_report(y_val, y_pred))

    # # Test on unseen testing data
    # X_test, _ = extract_lbp_features(test_images_dir)
    # y_test_pred = model.predict(X_test)

    # # Print results
    # print("\nTest Report:\n", classification_report([0]*len(X_test), y_test_pred))

    # Save model
    with open("./models/SVC_anti-spoof.joblib", "wb") as f:
        joblib.dump(model, f)

def predict():
    face_cascade = cv.CascadeClassifier(os.getenv("face_cascade"))
    model = joblib.load(open("./models/SVC_anti-spoof.joblib", "rb"))
    video_capture = cv.VideoCapture(0)
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=5, minSize=(100,100))
        for (x,y,w,h) in faces:
            lbp = extract_lbp_features_webcam(frame_gray).reshape(1, -1)
            # print(lbp)
            ypred = model.predict(lbp)
            # conf = int(max(model.predict_proba(ypred)[0]) * 100)
            if ypred == 1: pred = "real"
            elif ypred == 0: pred = "spoof"
            else: pred = "unknown"
            cv.putText(frame, str(f"{pred}  {ypred}"), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3, cv.LINE_AA)

        cv.imshow("Predictor", frame)
        cv.setWindowProperty("Predictor", cv.WND_PROP_TOPMOST, 1)
        # ESC to exit
        if cv.waitKey(1) & 0xff == 27:
         break
    
    video_capture.release()
    cv.destroyWindow("Predictor")

extract_lbp_features(train_images_dir)
train_SVC()
predict()