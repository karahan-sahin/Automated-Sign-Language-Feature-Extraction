import os
import re
import math
import requests
import cv2
from urllib.request import urlopen
from tqdm import tqdm
from lxml import html


from ..phonology.handshape import SignLanguagePhonology
from ..phonology.temporal import TemporalPhonology
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
from lib.sign_utils import *
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

INFO = {}

for v in tqdm(os.listdir('data/corpus/')[:10]):
    
    #camera = cv2.VideoCapture('data/corpus/'+ v)
    fvs = FileVideoStream('data/corpus/'+ v).start()
    
    fname = v[:-4]
        
    logs_left = None
    move_left = None
    logs_right = None
    move_right = None
    
    phonology_left = SignLanguagePhonology(hand='left')
    phonology_right = SignLanguagePhonology(hand='right')

    window_size = 5
    view_window = 100

    hand_relation = TemporalPhonology()
    
    frame = fvs.read()


    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        while fvs.more():

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = holistic.process(image)

            logs_right = phonology_right.getJointInfo(
                results.right_hand_landmarks, results.pose_landmarks, [ 'LOCATION', 'ORIENTATION', 'FINGER_SELECTION', 'MOVEMENT'])
            logs_left = phonology_left.getJointInfo(
                results.left_hand_landmarks, results.pose_landmarks, [ 'LOCATION', 'ORIENTATION', 'FINGER_SELECTION', 'MOVEMENT'])

            move_left = phonology_left.getMovementInformation(
                    view_window, window_size, ['Center_Diff', 'Center_MovAVG_x', 'Center_MovAVG_y', 'Center_MovAVG_z',
                                                'INDEX_FINGER', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY', 'SWITCH', 'LSE_x', 'LSE_y'])
            move_right = phonology_right.getMovementInformation(
                    view_window, window_size, ['Center_Diff', 'Center_MovAVG_x', 'Center_MovAVG_y', 'Center_MovAVG_z',
                                                'INDEX_FINGER', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY', 'SWITCH', 'LSE_x', 'LSE_y'])

            #hand_relation.findTemporalAlignment(
            #        phonology_left, phonology_right, view_window, window_size, metrics)

            #ss, frame = camera.read()
            frame = fvs.read()
            

            
    INFO[fname] = { }
    
    INFO[fname]['left_log'] = phonology_left.INFO_MEMORY
    INFO[fname]['left_move'] = move_left
    INFO[fname]['right_log'] = phonology_right.INFO_MEMORY
    INFO[fname]['right_move'] = move_right