import cv2 as cv
import numpy as np
from PIL import Image
from detect import detect
import base64
import io



def get_frame(img):

    frame = detect(img)
    ret, jpeg = cv.imencode('.jpg', frame)
    
    return jpeg.tobytes()