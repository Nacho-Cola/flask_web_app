import cv2 as cv
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

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

def detect(frame):

  with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

      frame = cv.flip(frame, 1)
      rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #BGR 이미지를 RGB로 변환
      img_h, img_w = frame.shape[:2]                   #img 크기
      results = face_mesh.process(rgb_frame)

      if results.multi_face_landmarks:

        mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
        (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
        (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)
            
            #홍체 테두리
        cv.circle(frame, center_left, int(l_radius), (0,255,0), 1, cv.LINE_AA)
        cv.circle(frame, center_right, int(r_radius), (0,255,0), 1, cv.LINE_AA) 

        dis_frame = frame


        mask = np.zeros((img_h,img_w), np.uint8)
        cv.circle(mask, center_left, int(l_radius), (255,0,0),-1, cv.LINE_AA)
        left = cv.bitwise_and(frame, frame, mask= mask)

        cv.circle(mask, center_right, int(r_radius), (255,0,0),-1, cv.LINE_AA)
        right = cv.bitwise_and(frame, frame, mask= mask)

        left_eye = left[center_left[1] - int(l_radius) : center_left[1] + int(l_radius) , center_left[0] - int(l_radius) : center_left[0]+ int(l_radius)]
        right_eye = right[center_right[1] - int(r_radius) : center_right[1] + int(r_radius) , center_right[0] - int(r_radius) : center_right[0]+ int(r_radius)]                 


        left_eye = cv.resize(left_eye, (10*int(l_radius), 10*int(l_radius)), interpolation = cv.INTER_LANCZOS4 )
        right_eye = cv.resize(right_eye, (10*int(r_radius), 10*int(r_radius)), interpolation = cv.INTER_LANCZOS4 )

        alpha = 1.
        left_eye= np.clip((1+alpha)*left_eye - 128*alpha, 0, 255).astype(np.uint8)
        right_eye= np.clip((1+alpha)*right_eye - 128*alpha, 0, 255).astype(np.uint8)

            #face crop
            
            
        face_detection_results = face_detection.process(frame[:,:,::-1])
            
        if face_detection_results.detections:

          for face_no, face in enumerate(face_detection_results.detections):

            face_data = face.location_data   #얼굴 좌표


        data = face_data.relative_bounding_box
        [h,w,c] = np.shape(frame)


        xleft = data.xmin*w
        xleft = int(xleft)
        xtop = data.ymin*h
        xtop = int(xtop)
        xright = data.width*w + xleft
        xright = int(xright)
        xbottom = data.height*h + xtop
        xbottom = int(xbottom)

        detected_faces = [(xleft, xtop, xright, xbottom)]

        for n, face_rect in enumerate(detected_faces):
            face = Image.fromarray(frame).crop(face_rect)
            face_np = np.asarray(face)
                
            L_results = model(left_eye)
            L_annotated_frame = L_results[0].plot()
            R_results = model(right_eye)
            R_annotated_frame = R_results[0].plot()
                
            if list(L_results[0].boxes.xywh) and list(R_results[0].boxes.xywh) :
                    # Visualize the results on the frame
              L_box_center_coor = L_results[0].boxes.xywh[0][:2]
              L_eye_center_coor = L_results[0].boxes.orig_shape /2
              L_error = L_box_center_coor - L_eye_center_coor
              L_error_n = L_error / L_eye_center_coor       
                   
              R_box_center_coor = R_results[0].boxes.xywh[0][:2]
              R_eye_center_coor = R_results[0].boxes.orig_shape /2
              R_error = R_box_center_coor - R_eye_center_coor
              R_error_n = R_error / R_eye_center_coor

                    
              if abs(R_error_n[0] - L_error_n[0])>0.2  or abs(R_error_n[1] - L_error_n[1])>0.2 :
                cv.imwrite(f'detected_face/detected_{tm.tm_hour}-{tm.tm_min}-{tm.tm_sec}.jpg', face_np)

                    
                # Display the annotated frame
            
  return dis_frame

def detect_img(frame):

  result = "undected"
  frame = cv.imread(frame)

  with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

      rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #BGR 이미지를 RGB로 변환
      img_h, img_w = frame.shape[:2]                   #img 크기
      results = face_mesh.process(rgb_frame)

      if results.multi_face_landmarks:

        mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
        (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
        (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)
            
            #홍체 테두리
        cv.circle(frame, center_left, int(l_radius), (0,255,0), 1, cv.LINE_AA)
        cv.circle(frame, center_right, int(r_radius), (0,255,0), 1, cv.LINE_AA) 

        mask = np.zeros((img_h,img_w), np.uint8)
        cv.circle(mask, center_left, int(l_radius), (255,0,0),-1, cv.LINE_AA)
        left = cv.bitwise_and(frame, frame, mask= mask)

        cv.circle(mask, center_right, int(r_radius), (255,0,0),-1, cv.LINE_AA)
        right = cv.bitwise_and(frame, frame, mask= mask)

        left_eye = left[center_left[1] - int(l_radius) : center_left[1] + int(l_radius) , center_left[0] - int(l_radius) : center_left[0]+ int(l_radius)]
        right_eye = right[center_right[1] - int(r_radius) : center_right[1] + int(r_radius) , center_right[0] - int(r_radius) : center_right[0]+ int(r_radius)]                 


        left_eye = cv.resize(left_eye, (10*int(l_radius), 10*int(l_radius)), interpolation = cv.INTER_LANCZOS4 )
        right_eye = cv.resize(right_eye, (10*int(r_radius), 10*int(r_radius)), interpolation = cv.INTER_LANCZOS4 )

        alpha = 1.
        left_eye= np.clip((1+alpha)*left_eye - 128*alpha, 0, 255).astype(np.uint8)
        right_eye= np.clip((1+alpha)*right_eye - 128*alpha, 0, 255).astype(np.uint8)

            #face crop
            
            
        face_detection_results = face_detection.process(frame[:,:,::-1])
            
        if face_detection_results.detections:

          for face_no, face in enumerate(face_detection_results.detections):

            face_data = face.location_data   #얼굴 좌표


        data = face_data.relative_bounding_box
        [h,w,c] = np.shape(frame)


        xleft = data.xmin*w
        xleft = int(xleft)
        xtop = data.ymin*h
        xtop = int(xtop)
        xright = data.width*w + xleft
        xright = int(xright)
        xbottom = data.height*h + xtop
        xbottom = int(xbottom)

        detected_faces = [(xleft, xtop, xright, xbottom)]

        for n, face_rect in enumerate(detected_faces):
            face = Image.fromarray(frame).crop(face_rect)
            face_np = np.asarray(face)
                
            L_results = model(left_eye)
            L_annotated_frame = L_results[0].plot()
            R_results = model(right_eye)
            R_annotated_frame = R_results[0].plot()
                
            if list(L_results[0].boxes.xywh) and list(R_results[0].boxes.xywh) :
                    # Visualize the results on the frame
              L_box_center_coor = L_results[0].boxes.xywh[0][:2]
              L_eye_center_coor = L_results[0].boxes.orig_shape /2
              L_error = L_box_center_coor - L_eye_center_coor
              L_error_n = L_error / L_eye_center_coor       
                   
              R_box_center_coor = R_results[0].boxes.xywh[0][:2]
              R_eye_center_coor = R_results[0].boxes.orig_shape /2
              R_error = R_box_center_coor - R_eye_center_coor
              R_error_n = R_error / R_eye_center_coor

                    
              if abs(R_error_n[0] - L_error_n[0])>0.2  or abs(R_error_n[1] - L_error_n[1])>0.2 :
          
                ret, jpeg = cv.imencode('.jpg', face_np)
                result = "detected"
                return jpeg, result

                    
                # Display the annotated frame
  ret, jpeg = cv.imencode('.jpg', frame)
  return jpeg, result