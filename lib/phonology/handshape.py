import os
import pickle
import pandas as pd
import numpy as np
import mediapipe as mp
from typing import List, Optional, Dict
from collections.abc import Sequence, Callable
from ..utils._math import getDistance, getCoordinates, getCosineAngle
from sklearn.preprocessing import minmax_scale
from lib.phonology.sign_utils import *
from collections import defaultdict
from typing import Dict, List

HandFeatures = Dict[str, int]

class HandshapePhonology():
    """The phonological information extraction from MediaPipe 
    
    
    """    
    
    def __init__(self, 
                 hand: str, 
                 normalize_angle = True,
                 normalize_orient = True,
                 normalize_location = True,
                 to_memory = True) -> None:
        """_summary_

        Args:
            hand (str): _description_
            normalize_angle (bool, optional): _description_. Defaults to True.
            normalize_orient (bool, optional): _description_. Defaults to True.
            normalize_location (bool, optional): _description_. Defaults to True.
            to_memory (bool, optional): _description_. Defaults to True.
        """

        if hand.capitalize() in [ 'Left', 'Right' ]: self.HAND = hand
        else: raise " ** The hand name must be either 'Left' or 'Right' ** "
        
        self.to_memory = to_memory
        
        self.normalize_angle = normalize_angle
        self.normalize_orient = normalize_orient
        self.normalize_location = normalize_location
        
        self.MEMORY = defaultdict(list)
        
        self.HAND_HISTORY = { }
        self.BODY_HISTORY = { }
        self.CONF_HISTORY = { }
        
        self.INFO = None
        
        self.CENTER_HISTORY = []
        self.APERTURE_HISTORY = []

    def getJointInfo(self, 
                     hand_pose: dict, 
                     body_pose: dict, 
                     selected: List[str]) -> HandFeatures:
        """Extracting static information from sign shape including:
            - Finger Selection/Coordination ( is_selected, is_curved, joint_angle )
            - Major/Minor Location
            - Orientation

        Args:
            hand_pose (dict): _description_
            body_pose (dict): _description_
            selected (list[str]): _description_

        Returns:
            HandFeatures: _description_
        """

        if hand_pose:
            # Find the center of the hand             
            CENTER = np.mean(list(map(lambda point: np.array([point.x, point.y, point.z]), hand_pose.landmark)) ,
                                axis=0)


            INFO = {}
            if 'FINGER_SELECTION' in selected: INFO.update(self.getFingerConfigurations(hand_pose.landmark))
            if 'ORIENTATION' in selected: INFO.update(self.getOrientation(hand_pose.landmark))
            
            SELECTION = [ getCoordinates(hand_pose.landmark[HAND_POINT2IDX[f'{joint}_TIP']]) for joint in[ 'THUMB',
                                                                            'INDEX_FINGER',
                                                                            'MIDDLE_FINGER',
                                                                            'RING_FINGER',
                                                                            'PINKY'] if INFO.get(f'{joint}_ANGLE_PIP_SELECT')  ]
            if SELECTION: CENTER = np.mean(SELECTION, axis=0)
            self.CENTER_HISTORY.append(CENTER)

            if 'LOCATION' in selected: INFO.update(self.getLocation(CENTER, body_pose.landmark))

            # Save it to the memory
            for feat, val in INFO.items(): self.MEMORY[feat].append(val)

            self.INFO = INFO
            
            return INFO

        return None
    
    # *************************************** #
    # **** FINGER SELECTION/CONFIGURATON **** #
    # *************************************** #

    def extractFingerInfo(self,
                          joint: str,
                          hand_pose: List[dict],
                          is_selected,
                          is_curved) -> HandFeatures:
        """_summary_

        Args:
            joint (str): _description_
            hand_pose (List[dict]): _description_
            is_selected (Callable[[int], bool]): _description_
            is_curved (Callable[[int, int], bool]): _description_
            normalize (bool): _description_

        Returns:
            HandFeatures: _description_
        """
        
        # Calculate angle via centering lower joint of a given finger
        PIP_IDX = HAND_POINT2IDX[f'{joint}_PIP'] if 'THUMB' != joint else HAND_POINT2IDX[f'{joint}_MCP']
        ANGLE_PIP = getCosineAngle(
            hand_pose[PIP_IDX-1],
            hand_pose[PIP_IDX],
            hand_pose[PIP_IDX+1],
        )
        
        # Calculate angle via centering lower joint of a given finger
        DIP_IDX = HAND_POINT2IDX[f'{joint}_DIP'] if 'THUMB' != joint else HAND_POINT2IDX[f'{joint}_IP']
        ANGLE_DIP = getCosineAngle(
            hand_pose[DIP_IDX-1],
            hand_pose[DIP_IDX],
            hand_pose[DIP_IDX+1],
        )
        
        # Calculate angle via centering wrist and lower joints as edges
        ANGLE_WRST_PIP = getCosineAngle(
            hand_pose[PIP_IDX-1],
            hand_pose[0],
            hand_pose[PIP_IDX+1],
        )
        
        # Calculate angle via centering wrist and higher joints as edges
        ANGLE_WRST_DIP = getCosineAngle(
            hand_pose[DIP_IDX-1],
            hand_pose[0],
            hand_pose[DIP_IDX+1],
        )
        
        return {
            
            #f'{joint}_ANGLE_PIP': ANGLE_PIP if not self.normalize_location else ANGLE_PIP / 180,
            f'{joint}_ANGLE_DIP': ANGLE_DIP if not self.normalize_location else ANGLE_DIP / 180,
            #f'{joint}_ANGLE_WRST_PIP': ANGLE_WRST_PIP if not self.normalize_location else ANGLE_WRST_PIP / 180,
            #f'{joint}_ANGLE_WRST_DIP': ANGLE_WRST_DIP if not self.normalize_location else ANGLE_WRST_DIP / 180,
            
            #f'{joint}_ANGLE_PIP_SELECT': is_selected(ANGLE_PIP),
            f'{joint}_ANGLE_DIP_SELECT': is_selected(ANGLE_DIP),
            #f'{joint}_ANGLE_WRST_PIP_SELECT': is_selected(ANGLE_WRST_PIP),
            #f'{joint}_ANGLE_WRST_DIP_SELECT': is_selected(ANGLE_WRST_DIP),         
            
            #f'{joint}_ANGLE_PIP_CURVE': is_curved(ANGLE_PIP),
            f'{joint}_ANGLE_DIP_CURVE': is_curved(ANGLE_DIP),
            #f'{joint}_ANGLE_WRST_PIP_CURVE': is_curved(ANGLE_WRST_PIP),
            #f'{joint}_ANGLE_WRST_DIP_CURVE': is_curved(ANGLE_WRST_DIP),    
        
        }

    def getFingerConfigurations(self,
                                hand_pose: List[dict],
                                min_angle=90,
                                max_curve=160) -> HandFeatures:
        """_summary_

        Args:
            hand_pose (List[dict]): _description_
            min_angle (int, optional): _description_. Defaults to 90.
            max_curve (int, optional): _description_. Defaults to 160.
            normalize (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        # Define functions for non-linear feature operation
        is_selected = lambda angle: True if angle >= min_angle else False
        is_curved = lambda angle: True if angle >= min_angle and max_curve <= angle else False
        
        INFO = {}
        INFO.update(self.extractFingerInfo('THUMB', hand_pose, is_selected, is_curved))
        INFO.update(self.extractFingerInfo('INDEX_FINGER', hand_pose, is_selected, is_curved))
        INFO.update(self.extractFingerInfo('MIDDLE_FINGER', hand_pose, is_selected, is_curved))
        INFO.update(self.extractFingerInfo('RING_FINGER', hand_pose, is_selected, is_curved))
        INFO.update(self.extractFingerInfo('PINKY', hand_pose, is_selected, is_curved))

        return INFO

    # *************************************** #
    # ************ ORIENTATION ************** #
    # *************************************** #
    
    def getOrientation(self, 
                       hand_pose: List[dict]) -> HandFeatures:
        """_summary_

        Args:
            hand_pose (List[dict]): _description_

        Returns:
            HandFeatures: _description_
        """


        INFO = {}
                
        # Extract PALM/BACK feature

        # Find x,y diff -> if left side is gets closer to right side meaning the hand rotates
        Y_HORIZONTAL = (hand_pose[HAND_POINT2IDX.get('THUMB_CMC')].x - hand_pose[HAND_POINT2IDX.get('PINKY_MCP')].x) + (hand_pose[HAND_POINT2IDX.get('THUMB_CMC')].y - hand_pose[HAND_POINT2IDX.get('PINKY_MCP')].y)
        # Normalize with true horizontal length
        HORIZONTAL_MAX = getDistance(getCoordinates(hand_pose[HAND_POINT2IDX.get('THUMB_CMC')]),getCoordinates(hand_pose[HAND_POINT2IDX.get('PINKY_MCP')]))
        
        # Differs between left and right hand
        if  Y_HORIZONTAL > 0.0: INFO['PALM/BACK_LABEL'] = 'palm' if self.HAND == 'Right' else 'back' 
        else: INFO['PALM/BACK_LABEL'] = 'back' if self.HAND == 'Right' else 'palm'
        INFO['PALM/BACK_SCORE'] = Y_HORIZONTAL if not self.normalize_orient else Y_HORIZONTAL / HORIZONTAL_MAX

        # Extract TIPS/WRIST feature

        # Find z diff -> vertical orient 
        Z_VERTICAL = hand_pose[HAND_POINT2IDX.get('WRIST')].z - hand_pose[HAND_POINT2IDX.get('MIDDLE_FINGER_TIP')].z
        # Normalize with true vertical length
        VERTICAL_MAX = getDistance(getCoordinates(hand_pose[HAND_POINT2IDX.get('WRIST')]),getCoordinates(hand_pose[HAND_POINT2IDX.get('MIDDLE_FINGER_TIP')])) 
            
        if  Z_VERTICAL > 0.0: INFO['TIPS/WRIST_LABEL'] = 'tips' 
        else: INFO['TIPS/WRIST_LABEL'] = 'wrist'
        INFO['TIPS/WRIST_SCORE'] = Z_VERTICAL if not self.normalize_orient else Z_VERTICAL / VERTICAL_MAX

        # Extract RADIAL/URNAL feature

        # Find z diff -> horizontal orient 
        Z_HORIZONTAL = hand_pose[HAND_POINT2IDX.get('THUMB_CMC')].z - hand_pose[HAND_POINT2IDX.get('PINKY_MCP')].z
        
        if  Z_VERTICAL > 0.0: INFO['RADIAL/URNAL_LABEL'] = 'urnal' 
        else: INFO['RADIAL/URNAL_LABEL'] = 'radial'
        INFO['RADIAL/URNAL_SCORE'] = Z_HORIZONTAL if not self.normalize_orient else Z_HORIZONTAL / HORIZONTAL_MAX

        return INFO

    # *************************************** #
    # ************* LOCATION **************** #
    # *************************************** #

    def getLocation(self, 
                    center: np.array, 
                    body_pose: List[dict]) -> HandFeatures:
        """_summary_

        Args:
            center (np.array): _description_
            body_pose (List[dict]): _description_

        Returns:
            HandFeatures: _description_
        """

        EYE_CENTER = getCoordinates(body_pose[BODY_POINT2IDX.get('LEFT_EYE_INNER')]) if getDistance(center, getCoordinates(body_pose[BODY_POINT2IDX.get('LEFT_EYE_INNER')])) < getDistance(center, getCoordinates(body_pose[BODY_POINT2IDX.get('RIGHT_EYE_INNER')])) else getCoordinates(body_pose[BODY_POINT2IDX.get('RIGHT_EYE_INNER')])
        EAR_CENTER = getCoordinates(body_pose[BODY_POINT2IDX.get('LEFT_EAR')]) if getDistance(center, getCoordinates(body_pose[BODY_POINT2IDX.get('LEFT_EAR')])) < getDistance(center, getCoordinates(body_pose[BODY_POINT2IDX.get('RIGHT_EAR')])) else getCoordinates(body_pose[BODY_POINT2IDX.get('RIGHT_EAR')])
        NOSE_CENTER = getCoordinates(body_pose[BODY_POINT2IDX.get('NOSE')])
        MOUTH_CENTER = np.mean([getCoordinates(body_pose[BODY_POINT2IDX.get('MOUTH_LEFT')]), getCoordinates(body_pose[BODY_POINT2IDX.get('MOUTH_RIGHT')])], axis=0)
        NECK_CENTER = np.mean([getCoordinates(body_pose[BODY_POINT2IDX.get('LEFT_SHOULDER')]), getCoordinates(body_pose[BODY_POINT2IDX.get('RIGHT_SHOULDER')])], axis=0)
        SHOULDER_CENTER = getCoordinates(body_pose[BODY_POINT2IDX.get('LEFT_SHOULDER')]) if getDistance(center, getCoordinates(body_pose[BODY_POINT2IDX.get('LEFT_SHOULDER')])) < getDistance(center, getCoordinates(body_pose[BODY_POINT2IDX.get('RIGHT_SHOULDER')])) else getCoordinates(body_pose[BODY_POINT2IDX.get('RIGHT_SHOULDER')])
        HIP_CENTER = np.mean([getCoordinates(body_pose[BODY_POINT2IDX.get('LEFT_HIP')]), getCoordinates(body_pose[BODY_POINT2IDX.get('LEFT_HIP')])], axis=0)
        CHEST_CENTER = np.mean([NECK_CENTER, HIP_CENTER], axis=0)
        ARM_CENTER = getCoordinates(body_pose[BODY_POINT2IDX.get('LEFT_ELBOW')]) if getDistance(center, getCoordinates(body_pose[BODY_POINT2IDX.get('LEFT_ELBOW')])) < getDistance(center, getCoordinates(body_pose[BODY_POINT2IDX.get('RIGHT_ELBOW')])) else getCoordinates(body_pose[BODY_POINT2IDX.get('RIGHT_ELBOW')])

        POSITIONS = [
               getDistance(center, EYE_CENTER), 
               getDistance(center, EAR_CENTER), 
               getDistance(center, NOSE_CENTER), 
               getDistance(center, MOUTH_CENTER), 
               getDistance(center, SHOULDER_CENTER),
               getDistance(center, CHEST_CENTER),
               getDistance(center, HIP_CENTER),
               getDistance(center, ARM_CENTER),
        ]
        
        if self.normalize_location: POSITIONS = minmax_scale(POSITIONS)
        
        INFO = {
            'EYE': POSITIONS[0],
            'EAR': POSITIONS[1],
            'NOSE': POSITIONS[2],
            'MOUTH': POSITIONS[3], 
            'SHOULDER': POSITIONS[4],
            'CHEST': POSITIONS[5],
            'HIP': POSITIONS[6],
            'ARM': POSITIONS[7],
        }

        return dict(sorted(INFO.items(), key=lambda x: x[1]))

