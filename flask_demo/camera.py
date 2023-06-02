import cv2 as cv
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from detect import detect

from time import time
import time 
tm =time.localtime()

model = YOLO('best.pt')

count = 0

mp_face_mesh = mp.solutions.face_mesh

mp_face_detection = mp.solutions.face_detection
 
# Setup the face detection function.
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
 
# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils

#눈, 홍채 좌표
LEFT_EYE=[362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398 ] 
 # right eyes indices 
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

LEFT_IRIS = [474, 475, 476, 477] 
RIGHT_IRIS = [469, 470, 471, 472]


class VideoCamera(object):
    def __init__(self):
        self.video = cv.VideoCapture(0)

    def __del__(self):
        self.video.release()        

    def get_frame(self):
        ret, frame = self.video.read()
        frame = detect(frame)
        ret, jpeg = cv.imencode('.jpg', frame)
            
        return jpeg.tobytes()