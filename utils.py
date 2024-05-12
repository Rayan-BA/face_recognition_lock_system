import os
from shutil import rmtree
import cv2 as cv
import numpy as np
from uuid import uuid1
from tqdm import tqdm
import random

face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

def _load_images(sub_dir):
    faces = []
    for img in tqdm(os.listdir(sub_dir)):
        path = sub_dir + "/" + img
        face = cv.imread(path)
        face = cv.resize(face, (160, 160))
        faces.append(face)
    return faces

def load_data(dir: str):
    print(" [INFO] Loading classes...")
    x, y = [], []
    for sub_dir in os.listdir(dir):
        path = dir + "/" + sub_dir
        faces = _load_images(path)
        labels = [sub_dir for _ in range(len(faces))]
        x.extend(faces)
        y.extend(labels)
    print(" [INFO] Loading done.")
    return np.asarray(x), np.asarray(y)

def _create_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

def _removeDir(dir_name):
    if os.path.isdir(dir_name):
        rmtree(dir_name)

def _save_face(face, path):
    random_id = uuid1().int >> 102 # uuid is 128 bits, shift right by 102 bits to shorten
    _create_dir(path)
    cv.imwrite(f"{path}/{random_id}.jpg", face)

def _draw_prog_bar(frame, w, h, prog):
    padding = 20
    prog_bar_start = padding
    prog_bar_end = w-padding
    cv.rectangle(frame, (prog_bar_start, h-30), (prog_bar_end, h-10), (0,0,255), 2)
    prog_bar = int(prog * prog_bar_end/100)
    if prog_bar > padding:
        cv.rectangle(frame, (prog_bar_start, h-30), (prog_bar, h-10), (0,0,255), -1) # thickness -1 for fill

def collect_samples(dataset_path, mode, pic_limit=150):
    label = ""
    if mode == "face_rec":
        label = input(" [INPUT] Enter a label: ")
    elif mode == "anti-spoof":
        while label != "real" and label != "spoof":
            print(""" [SELECT]
                  
            Select label:
            1) real
            2) spoof
            
            Press any other key to exit.""")
            op = input(" Your selection: ")
            match op:
                case "1": label = "real"
                case "2": label = "spoof"
    cam = cv.VideoCapture(0)
    counter = 0
    limit = pic_limit
    frame_width  = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
    padding = 10
    while cam.isOpened():
        _, frame = cam.read()
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(100,100))
        for (x,y,w,h) in faces:
            counter += 1
            face = frame[y:y+h, x:x+w]
            cv.rectangle(frame, (x-padding, y-padding), (x+w+padding, y+h+padding), (0,0,255), 2)
        try: _save_face(face, f"{dataset_path}/{label}")
        except: pass
        _draw_prog_bar(frame, frame_width, frame_height, int((counter/limit)*100))
        cv.imshow("Webcam Capture", frame)
        cv.setWindowProperty("Webcam Capture", cv.WND_PROP_TOPMOST, 1)
        if cv.waitKey(1) & 0xff == 27 or counter >= limit: # ESC to exit
            break
    cam.release()
    cv.destroyWindow("Webcam Capture")

def collect_samples_onpress(dataset_path, mode):
    label = ""
    if mode == "face_rec":
        label = input(" [INPUT] Enter a label: ")
    elif mode == "anti-spoof":
        while label != "real" and label != "spoof":
            print(""" [SELECT]
                  
            Select label:
            1) real
            2) spoof
            
            Press any other key to exit.""")
            op = input(" Your selection: ")
            match op:
                case "1": label = "real"
                case "2": label = "spoof"
    cam = cv.VideoCapture(0)
    padding = 10
    while cam.isOpened():
        _, frame = cam.read()
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(100,100))
        for (x,y,w,h) in faces:
            face = frame[y:y+h, x:x+w]
            cv.rectangle(frame, (x-padding, y-padding), (x+w+padding, y+h+padding), (0,0,255), 2)
        try:
            _save_face(face, f"{dataset_path}/{label}")
        except: pass
        cv.imshow("Webcam Capture", frame)
        cv.setWindowProperty("Webcam Capture", cv.WND_PROP_TOPMOST, 1)
        if cv.waitKey(1) & 0xff == 27: # ESC to exit
            break
    cam.release()
    cv.destroyWindow("Webcam Capture")

def augment(input, output, limit=None):
    path = os.listdir(input)
    for i, img_file in enumerate(path):
        if limit is not None and i >= limit: break
        image = cv.imread(input + "/" + img_file)
        rows, cols, channels = image.shape
        D = random.uniform(-10, 10)
        M = cv.getRotationMatrix2D((cols/2, rows/2), D, 1)
        image = cv.warpAffine(image, M, (cols, rows))
        if random.randint(0,1):
            image = cv.flip(image, 1)
        if random.randint(0,1):
            bright = np.ones(image.shape, dtype="uint8") * random.randrange(50, 71)
            image = cv.add(image, bright)
        if random.randint(0,1):
            bright = np.ones(image.shape, dtype="uint8") * random.randrange(50, 71)
            image = cv.subtract(image, bright)
        if not os.path.isdir(output):
            os.mkdir(output)
        cv.imwrite(f"{output}/{uuid1().int >> 100}.jpg", image)

def add_real(vid_file):
    cap = cv.VideoCapture(vid_file)
    count = 0
    while count < 50:
        _, frame = cap.read()
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5, minSize=(100,100))
        for (x,y,w,h) in faces:
            face = frame[y:y+h, x:x+w]
        cv.imwrite(f"./antispoof-dataset/real/{uuid1().int >> 100}.jpg", face)
        count += 1
    cap.release()

def add_spoof(vid_file):
    cap = cv.VideoCapture(vid_file)
    count = 0
    while count < 50:
        _, frame = cap.read()
        cv.imwrite(f"./antispoof-dataset/spoof/{uuid1().int >> 100}.jpg", frame)
        count += 1
    cap.release()

def extract_lbp_features(image, radius=1, num_points=8, eps=1e-7):
    from skimage.feature import local_binary_pattern
    
    lbp = local_binary_pattern(image, num_points, radius, method="uniform")

    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))

    hist = hist.astype("float")
    hist /= (hist.sum() + eps)

    return hist

if __name__ == "__main__":
    pass
    # augment("./tmp/Ray", "./augmentedRay")