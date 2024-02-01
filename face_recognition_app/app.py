# import cv2 as cv
# from time import time

import utils
import LBPHFaceRecognizer

def main():
    print(" [INFO] Program started")
    print(""" [SELECT]
            1) Detect Faces
            2) Collect Face Samples
            3) Train Recognizer
            4) Recognize Faces
            
            Press any other key to exit.""")
    while True:
        op = input("Your selection: ")
        match op:
            case "1": utils.detectFace()
            case "2": utils.collectFaceSamples()
            case "3": LBPHFaceRecognizer.train()
            case "4": LBPHFaceRecognizer.recognize()
            case _: break
    print(" [INFO] Program ended")

main()
