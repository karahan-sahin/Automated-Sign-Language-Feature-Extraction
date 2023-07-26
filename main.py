import os
import cv2
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np
import pandas as pd
from sign_utils import *
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

from scipy.fft import fft, fftfreq

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

st.title("Sign Phonological Feature Detection")
#st.image('images/hand-landmarks.png')

FRAME_WINDOW = st.image([])


videos = [0] + ['data/'+file_name for file_name in os.listdir('data')]

option = st.selectbox('Select vids', videos )

camera = cv2.VideoCapture(option)
#camera = cv2.VideoCapture(0)
run = st.button('Run')

while run:
    _, frame = camera.read()
    
    phonology = SignLanguagePhonology()    
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
      
      display = st.selectbox( 'Select metric', ( 'feature', 'boundary' ), index=1 )
      
      if display == 'feature': selected = st.multiselect( 'Select metric', ( 'LOCATION', 'ORIENTATION', 'FINGER_SELECTION', 'MOVEMENT' ), default=['MOVEMENT'] )
      if display == 'boundary':
        selected = []
        window_size = st.number_input('Select MovAvg window',value=5)
        view_window = st.number_input('Select Motion Track window',value=100)
        
        #value = st.number_input('Select window',value=-3)
        #tolerance = st.number_input('Select window',value=50)
      
      with st.empty():
        while camera.isOpened():
          
            ret, frame = camera.read()
            
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Make Detections
            results = holistic.process(image)
            
            # Recolor image back to BGR for rendering
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
      
            # Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # Face
            # TODO: ADD NON-MANUAL MARKERS
            # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
      
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      
            FRAME_WINDOW.image(image)
  
            logs = phonology.getJointInfo(results, selected)
            
            # TODO: JOINT APERTURE DIFFERENCE
            
            
            if display == 'boundary':
              
              Center_Diff = phonology.DIFF[-view_window:]
              
              Center_MovAVG_x = pd.Series(np.array(phonology.CENTER[-view_window:])[:,0]).rolling(window_size).mean().tolist() 
              Center_MovAVG_y = pd.Series(np.array(phonology.CENTER[-view_window:])[:,1]).rolling(window_size).mean().tolist() 
              Center_MovAVG_z = pd.Series(np.array(phonology.CENTER[-view_window:])[:,2]).rolling(window_size).mean().tolist() 
              
              SWITCH = phonology.START[-view_window:]
              
              min_len = min(len(Center_Diff),len(Center_MovAVG_x),len(SWITCH))
              
              st.line_chart(pd.DataFrame({
                
                'Center_Diff': Center_Diff[:min_len],
                
                'Center_MovAVG_x': Center_MovAVG_x[:min_len],
                'Center_MovAVG_y': Center_MovAVG_y[:min_len],
                'Center_MovAVG_z': Center_MovAVG_z[:min_len],
                
                #'Center_MovAVG_x': is_periodic(Center_MovAVG_x[:min_len], value, tolerance/100),
                #'Center_MovAVG_y': is_periodic(Center_MovAVG_y[:min_len], value, tolerance/100),
                #'Center_MovAVG_z': is_periodic(Center_MovAVG_z[:min_len], value, tolerance/100),
                              
                'SWITCH': SWITCH[:min_len]
              }))
              
            else: st.write(logs)
            
            if option:
              time.sleep(0.2)
          

else:
    camera.release()
    cv2.destroyAllWindows()