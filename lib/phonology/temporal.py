import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lib.phonology.handshape import HandshapePhonology
from sklearn.metrics.pairwise import cosine_similarity

class TemporalPhonology():

    def __init__(self, selected) -> None:
        
        self.selected = selected
        self.HISTORY = []
        self.THRESHOLD = 0.1


    def getTemporalInformation(self, left, right):
        
        INFO = {}
        
        if left.INFO: INFO.update({'LEFT_'+info: val for info, val in left.INFO.items()})
        if right.INFO: INFO.update({'RIGHT_'+info: val for info, val in right.INFO.items()})
    
        if 'Path' in self.selected: INFO.update(self.getPathMannerFeatures(left, right))
        if 'Spacial' in self.selected: INFO.update(self.getSpacialMannerFeatures(left, right))
        if 'Temporal' in self.selected: INFO.update(self.getTemporalMannerFeatures(left, right))
        if 'Setting' in self.selected: INFO.update(self.getPlaceFeature(left, right))
        
        self.HISTORY.append(INFO)
    
        return INFO

    # *************************************** #
    # ************ PATH MANNER ************** #
    # *************************************** #
        
    def getPathMannerFeatures(self, 
                              left: HandshapePhonology, 
                              right: HandshapePhonology,
                              window=100):
        
        
        INFO = {}
        
        if left.INFO and len(left.CENTER_HISTORY) > 10:
            try:
                def func(x, a, b, c): return a * np.exp(-b * x) + c
                popt, pcov = curve_fit(func, np.array(left.CENTER_HISTORY[-window:])[:,0],  np.array(left.CENTER_HISTORY[-window:])[:,1])
                # Check the the gradient of first variable to determine the curve
                INFO['LEFT_CURVE_LABEL'] = 'straight' if popt[0] >= self.THRESHOLD else 'arc'
                INFO['LEFT_CURVE_SCORE'] = popt[0]
            except: pass
        
        if right.INFO and len(right.CENTER_HISTORY) > 20:
            try:
                def func(x, a, b, c): return a * np.exp(-b * x) + c
                popt, pcov = curve_fit(func, np.array(right.CENTER_HISTORY[-window:])[:,0],  np.array(right.CENTER_HISTORY[-window:])[:,1])
                # Check the the gradient of first variable to determine the curve
                INFO['RIGHT_CURVE_LABEL'] = 'straight' if popt[0] >= self.THRESHOLD else 'arc'
                INFO['RIGHT_CURVE_SCORE'] = popt[0]
            except: pass
        
        return INFO
    
    # *************************************** #
    # ********** TEMPORAL MANNER ************ #
    # *************************************** #

    
    def getTemporalMannerFeatures(self,
                                  left: HandshapePhonology,
                                  right: HandshapePhonology,
                                  window=5):
        
        INFO = {}
        if right.INFO and len(right.CENTER_HISTORY) > window*2:
            from numpy import dot
            from numpy.linalg import norm
            
            a = np.array(right.CENTER_HISTORY[-window*2: -window])
            b = np.array(right.CENTER_HISTORY[-window:])
            cos_sim_x = cosine_similarity(a,b)

            a = np.array(right.CENTER_HISTORY[-window*2: -window])
            b = np.array(right.CENTER_HISTORY[-window:])
            cos_sim_y = cosine_similarity(a,b)
                        
            mean_diff = np.mean([cos_sim_x, cos_sim_y])
            INFO['RIGHT_REPETITION_LABEL'] = True if mean_diff > 0.7 else False
            INFO['RIGHT_REPETITION_SCORE'] = mean_diff
            
            
        INFO = {}
        if left.INFO and len(left.CENTER_HISTORY) > window*2:
            from numpy import dot
            from numpy.linalg import norm
            
            a = np.array(left.CENTER_HISTORY[-window*2: -window])
            b = np.array(left.CENTER_HISTORY[-window:])
            cos_sim_x = cosine_similarity(a,b)

            a = np.array(left.CENTER_HISTORY[-window*2: -window])
            b = np.array(left.CENTER_HISTORY[-window:])
            cos_sim_y = cosine_similarity(a,b)
                        
            mean_diff = np.mean([cos_sim_x, cos_sim_y])
            INFO['LEFT_REPETITION_LABEL'] = True if mean_diff > 0.99 else False
            INFO['LEFT_REPETITION_SCORE'] = mean_diff
        
        return INFO
    
    # *************************************** #
    # *********** SPACIAL MANNER ************ #
    # *************************************** #
    
    def getSpacialMannerFeatures(self, left, right):
        
        INFO = {}
        
        if left.INFO and right.INFO and len(left.CENTER_HISTORY) > 2 and len(right.CENTER_HISTORY) > 2:
            # IN FRONT
            Z_DIFF = left.CENTER_HISTORY[-1][2] - right.CENTER_HISTORY[-1][2]
            INFO['IN_FRONT_LABEL'] = 'left' if Z_DIFF > 0.0 else 'right'
            INFO['IN_FRONT_SCORE'] = Z_DIFF
            
            # ABOVE/BELOW
            Y_DIFF = left.CENTER_HISTORY[-1][1] - right.CENTER_HISTORY[-1][1]
            INFO['ABOVE/BELOW_LABEL'] = 'left' if Y_DIFF > 0.0 else 'right'
            INFO['ABOVE/BELOW_SCORE'] = Y_DIFF
        
            # INTERVOWEN
            #Z_DIFF = 
            #getCoordinate(left.hand_pose[sign_utils.HAND_POINT2IDX('WRIST')] )
            #right.CENTER_HISTORY[1]
            #getCoordinate(left.hand_pose[sign_utils.HAND_POINT2IDX('INDEX_FINGER_TIP')] )
            #INFO['INTERVOWEN_LABEL'] = Z_DIFF > 0.0 
            #INFO['INTERVOWEN_SCORE'] = Z_DIFF
        
        return INFO
    
    # *************************************** #
    # *********** PLACE SETTING ************* #
    # *************************************** #
    
    def getPlaceFeature(self, left, right):

        INFO = {}

        if left.INFO and len(left.CENTER_HISTORY) > 2:
            # SETTING
            INFO['LEFT_SETTING_LABEL'] = 'high' if left.CENTER_HISTORY[-2][1] > left.CENTER_HISTORY[-1][1] else 'low'
            INFO['LEFT_SETTING_SCORE'] = left.CENTER_HISTORY[-2][1] - left.CENTER_HISTORY[-1][1]
            
        if right.INFO and len(right.CENTER_HISTORY) > 2:
            INFO['RIGHT_SETTING_LABEL'] = 'high' if right.CENTER_HISTORY[-2][1] > right.CENTER_HISTORY[-1][1] else 'low'
            INFO['RIGHT_SETTING_SCORE'] = right.CENTER_HISTORY[-2][1] - right.CENTER_HISTORY[-1][1]

        # APERTURE
        
        if left.INFO and len(left.APERTURE_HISTORY) > 2:
            INFO['LEFT_APERTURE_LABEL'] = 'opening' if sum(left.APERTURE_HISTORY[-1] - left.APERTURE_HISTORY[-2]) > 0 else 'closing'
            INFO['LEFT_APERTURE_SCORE'] = sum(left.APERTURE_HISTORY[-1] - left.APERTURE_HISTORY[-2])

        if right.INFO and len(right.APERTURE_HISTORY) > 2:
            INFO['RIGHT_APERTURE_HISTORY_LABEL'] = 'opening' if sum(right.APERTURE_HISTORY[-1] - right.APERTURE_HISTORY[-2]) > 0 else 'closing'
            INFO['RIGHT_APERTURE_SCORE'] = sum(right.APERTURE_HISTORY[-1] - right.APERTURE_HISTORY[-2])

        
        return INFO