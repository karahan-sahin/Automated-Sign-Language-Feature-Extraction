import os
import cv2
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sign_utils import *
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

from scipy.fft import fft, fftfreq

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

st.set_page_config(layout="wide")

st.title("Sign Phonological Feature Detection")
# st.image('images/hand-landmarks.png')

FRAME_WINDOW = st.sidebar.image([])

videos = [0] + ['data/'+file_name for file_name in os.listdir('data')]

option = st.sidebar.selectbox('Select vids', videos)

camera = cv2.VideoCapture(option)

display = st.sidebar.selectbox(
    'Select metric', ('feature', 'boundary'), index=1)

if display == 'feature':
    selected = st.sidebar.multiselect(
        'Select metric', ('LOCATION', 'ORIENTATION', 'FINGER_SELECTION', 'MOVEMENT'), default=['MOVEMENT'])

if display == 'boundary':

    selected = []
    window_size = st.sidebar.number_input(
        'Select MovAvg window', value=5)
    view_window = st.sidebar.number_input(
        'Select Motion Track window', value=100)

    hand = st.sidebar.selectbox(
        'Select hand for movement tracking', ( 'BOTH', 'LEFT', 'RIGHT' ))

    if not hand == 'BOTH':
        metrics = st.sidebar.multiselect(label='Select Metrics',
                                        options=['Center_Diff', 'Center_MovAVG_x', 'Center_MovAVG_y', 'Center_MovAVG_z',
                                                'INDEX_FINGER', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY', 'SWITCH', 'LSE_x', 'LSE_y'],
                                        default=['Center_Diff', 'Center_MovAVG_x', 'Center_MovAVG_y', 'Center_MovAVG_z', 'INDEX_FINGER', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY', 'SWITCH', 'LSE_x', 'LSE_y'])

    else:
        metrics = st.sidebar.multiselect(label='Select Metrics',
                                        options=['Center_MovAVG_Left', 'Center_MovAVG_Right', 'Aperture_MovAVG_Left', 'Aperture_MovAVG_Right', 'SWITCH_Left', 'SWITCH_Right'],
                                        default=['Center_MovAVG_Left', 'Center_MovAVG_Right', 'Aperture_MovAVG_Left', 'Aperture_MovAVG_Right', 'SWITCH_Left', 'SWITCH_Right'])


run = st.sidebar.button('Run')

while run:

    _, frame = camera.read()

    phonology_left = SignLanguagePhonology(hand='left')
    phonology_right = SignLanguagePhonology(hand='right')

    hand_relation = TwoHandedPhonology()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        # value = st.number_input('Select window',value=-3)
        # tolerance = st.number_input('Select window',value=50)

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
                mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # Left Hand
                mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # Pose Detections
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                FRAME_WINDOW.image(image)

                logs_right = phonology_right.getJointInfo(
                    results.right_hand_landmarks, results.pose_landmarks, selected)
                logs_left = phonology_left.getJointInfo(
                    results.left_hand_landmarks, results.pose_landmarks, selected)

                if display == 'boundary':
                    # FIXME: MULTI-HAND PLOT
                    if hand == 'LEFT':
                        st.line_chart(phonology_left.getMovementInformation(
                            view_window, window_size, metrics))
                    if hand == 'RIGHT':
                        st.line_chart(phonology_right.getMovementInformation(
                            view_window, window_size, metrics))

                    if hand == 'BOTH':
                        st.line_chart(hand_relation.findTemporalAlignment(
                            phonology_left, phonology_right, view_window, window_size, metrics))

                else:
                    if logs_right or logs_left:

                        # if phonology_left.SWITCH or phonology_right.SWITCH:
                        #    pass

                        cols = st.columns(len(selected))
                        for idx, (col_name, col) in enumerate(zip(selected, cols)):
                            col.markdown(f'### {col_name}')

                            INFO = {}
                            if logs_right:
                                INFO['RIGHT'] = logs_right[idx]
                            if logs_left:
                                INFO['LEFT'] = logs_left[idx]

                            col.write(INFO)
                    else:
                        st.write('Hand Not Found')

                if option:
                    time.sleep(0.02)


else:
    camera.release()
    cv2.destroyAllWindows()
