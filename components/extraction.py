import os
import sys
import cv2
import asyncio
import streamlit as st
import mediapipe as mp
import hydralit_components as hc
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
from lib.phonology.handshape import *
from lib.phonology.temporal import *
from lib.phonology.classifier import *
from lib.phonology.elan import ELANWriter
from components.navbar import navbar
from datetime import datetime


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def extract():

    FRAME_WINDOW = st.sidebar.image([])
    PARAMS = navbar('feature-extraction')

    LEFT = HandshapePhonology('Left')
    RIGHT = HandshapePhonology('Right')
    TEMPORAL = TemporalPhonology(PARAMS['selected'])
    if PARAMS['display'] == 'classification':CLASSIFIER = LexicalClassification(PARAMS)    

    idx = 0
    while PARAMS['run'] and not PARAMS['stop']:
        
        try:
            _, frame = PARAMS['camera'].read()

            with mp_holistic.Holistic(min_detection_confidence=0.1, min_tracking_confidence=0.1) as holistic:

                with st.empty():

                    while PARAMS['camera'].isOpened():

                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = holistic.process(image)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                        # handshape_recognition_result = recognizer_handshape.recognize(mp_image)
                        # location_recognition_result = recognizer_location.recognize(mp_image)
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
                            
                        if PARAMS['display'] == 'classification':
                            # if handshape_recognition_result.gestures:
                            #     cat =handshape_recognition_result.gestures[0][0].category_name
                            #     if cat:
                            #         image = Image.open('images/handshapes/'+cat+'.png')
                            #         #st.image(image, caption=cat)
                            #         #st.write(handshape_recognition_result) 
                            # if location_recognition_result.gestures:
                            #     cat =location_recognition_result.gestures[0][0].category_name
                            #     if cat:
                            #         image = Image.open('images/location/'+cat+'.png')
                            #         st.image(image, caption=cat)
                            #         #st.write(location_recognition_result) 
                            pass
                                    
                            #if CLASSIFIER.LEXICON_MEMORY: st.dataframe(pd.DataFrame.from_records(CLASSIFIER.LEXICON_MEMORY[-20:]), height=750, use_container_width=True)
                            #asyncio.run(CLASSIFIER.predict(CLASSIFIER.transform(TEMPORAL.HISTORY), topK=PARAMS['topK']))

                        ret, frame = PARAMS['camera'].read()

                        if idx % 10 == 0 or not ret:
                            if PARAMS['display'] == 'feature':
                                st.session_state.log = TEMPORAL.HISTORY
                            if PARAMS['display'] == 'classification':
                                st.session_state.log = CLASSIFIER.LEXICON_MEMORY
                        
                        idx+=1
                            
        except cv2.error:
            PARAMS['run'] = False
            PARAMS['camera'].release()
            cv2.destroyAllWindows()


    if st.session_state.log:
        fname = st.text_input('File Name')
        save = st.button('Save Output')
        
        if save:

            if PARAMS['export_type'] == 'Excel File':
                if 'file_name' in PARAMS.keys():
                    pd.DataFrame.from_records(st.session_state.log).to_csv(f'data/output/{fname}.csv', index=False)
                    st.toast(f'Features are saved to data/output/{fname}.csv')
                else:
                    pd.DataFrame.from_records(st.session_state.log).to_csv(f'data/output/{fname}.csv', index=False)
                    st.toast(f'Features are saved to data/output/{fname}.csv')
            
            if PARAMS['export_type'] == 'Eaf File': 
                
                print(' ** Writing File **')
                
                EL = ELANWriter(
                    feature_df=pd.DataFrame.from_records(st.session_state.log),
                    video_path=PARAMS['option'],
                    percentage=PARAMS['percentage'],
                    roll_window=PARAMS['roll'],
                    threshold=PARAMS['threshold'],
                )
                
                EL.extractFeatures(
                    output_file='outputs/elan/'+fname
                )
                
                st.toast(f'Features are written on ELAN to outputs/elan/{fname}.eaf')
                
                
                print(' ** Written File **')