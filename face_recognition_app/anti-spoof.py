from dotenv import load_dotenv
import os
import cv2 as cv
from uuid import uuid1

# TODO: continue

load_dotenv()
kaggle_antispoof_dataset = os.getenv("kaggle_antispoof_dataset")
antispoof_dataset = os.getenv("antispoof_dataset")
face_cascade = cv.CascadeClassifier(os.getenv("face_cascade"))

def create_dataset_struct():
    try:
        os.mkdir(antispoof_dataset)
        os.mkdir(f"{antispoof_dataset}/real")
        os.mkdir(f"{antispoof_dataset}/spoof")
    except:
        pass

def populate_dataset():
    for sub_dir in os.listdir(kaggle_antispoof_dataset):
        sub_path = f"{kaggle_antispoof_dataset}/{sub_dir}"
        for vid_file in os.listdir(sub_path):
            sd = sub_path.split("/")[-1]
            vid =  cv.VideoCapture(f"{sub_path}/{vid_file}")
            process_frame = True
            while not vid.isOpened():
                vid =  cv.VideoCapture(f"{sub_path}/{vid_file}")
                cv.waitKey(100)
            while True:
                ret, frame = vid.read()
                gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                face = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=7, minSize=(100,100))
                # print(d)
                # exit()
                if ret and process_frame:
                    for (x, y, w, h) in face:
                        pos_frame = vid.get(cv.CAP_PROP_POS_FRAMES)
                        face = frame[y:y+h, x:x+w]
                        face = cv.resize(face, (160,160))
                        if sd == "real":
                            cv.imwrite(f"{antispoof_dataset}/real/{uuid1().int >> 102}.jpg", face)
                        else:
                            cv.imwrite(f"{antispoof_dataset}/spoof/{uuid1().int >> 102}.jpg", face)
                elif not ret:
                    vid.set(cv.CAP_PROP_POS_FRAMES, pos_frame - 1)
                    cv.waitKey(100)
                process_frame = not process_frame
                if cv.waitKey(1) & 0xff == 27:
                    break
                if vid.get(cv.CAP_PROP_POS_FRAMES) == vid.get(cv.CAP_PROP_FRAME_COUNT):
                    break
    vid.release()
    cv.destroyAllWindows()

create_dataset_struct()
populate_dataset()