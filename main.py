import sys
import os
import cv2
import streamlit as st
import mediapipe as mp
from lib.phonology.handshape import *
from lib.phonology.temporal import *
from components.navbar import navbar

if 'log' not in st.session_state:
    st.session_state.log = None

st.set_page_config(layout="wide")

st.title("Sign Phonological Feature Detection")

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

FRAME_WINDOW = st.sidebar.image([])

PARAMS = navbar()

LEFT = HandshapePhonology('Left')
RIGHT = HandshapePhonology('Right')
TEMPORAL = TemporalPhonology(PARAMS['selected'])

export = False
stop = st.sidebar.button('Save Live Stream', key=2)

idx = 0
while PARAMS['run'] and not stop:
    
    try:
        _, frame = PARAMS['camera'].read()

        with mp_holistic.Holistic(min_detection_confidence=0.1, min_tracking_confidence=0.1) as holistic:

            with st.empty():

                while PARAMS['camera'].isOpened():

                    ret, frame = PARAMS['camera'].read()
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    FRAME_WINDOW.image(image)
                    
                    RIGHT.getJointInfo(results.right_hand_landmarks, results.pose_landmarks, PARAMS['selected'])
                    LEFT.getJointInfo(results.left_hand_landmarks, results.pose_landmarks, PARAMS['selected'])
                    INFO = TEMPORAL.getTemporalInformation(LEFT, RIGHT)
                                        
                    if PARAMS['display'] == 'feature':
                        if INFO: st.dataframe(pd.DataFrame.from_records(TEMPORAL.HISTORY[-20:]), height=750)
                    if PARAMS['display'] == 'boundary':
                        st.line_chart(RIGHT.CENTER_HISTORY)
                        
                    if idx % 10 == 0:
                        st.session_state.log = TEMPORAL.HISTORY
                        
                    idx+=1
                    
    except cv2.error:
        PARAMS['run'] = False
        PARAMS['camera'].release()
        cv2.destroyAllWindows()


if st.session_state.log:
    from datetime import datetime
    fname = st.text_input('File Name')
    save = st.button('Save Output')
    if save:
        if 'file_name' in PARAMS.keys():
            pd.DataFrame.from_records(st.session_state.log).to_csv(f'data/output/{fname}.csv', index=False)
            st.toast(f'Features are saved to data/output/{fname}.csv')
        else:
            pd.DataFrame.from_records(st.session_state.log).to_csv(f'data/output/{fname}.csv', index=False)
            st.toast(f'Features are saved to data/output/{fname}.csv')

