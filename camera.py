import cv2 as cv
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from detect import detect

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