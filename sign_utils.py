import os
import pickle
import pandas as pd
import numpy as np
from scipy.spatial import distance
from fastdtw import fastdtw

from scipy.linalg import lstsq

from tqdm import tqdm
import math
import pickle
import os
from collections import defaultdict

KEY_SET = set(['CHEST_PROXIMITY_RIGHT','MOUTH_PROXIMITY_RIGHT','NOSE_PROXIMITY_RIGHT','EAR_PROXIMITY_RIGHT','EYE_PROXIMITY_RIGHT','PALM/BACK_PROXIMITY_RIGHT','TIPS/WRIST_PROXIMITY_RIGHT','URNAL/RADIAL_PROXIMITY_RIGHT','THUMB_ANGLE_RIGHT','THUMB_SELECTION_RIGHT','THUMB_CURVED_RIGHT','INDEX_FINGER_ANGLE_RIGHT','INDEX_FINGER_SELECTION_RIGHT','INDEX_FINGER_CURVED_RIGHT','MIDDLE_FINGER_ANGLE_RIGHT','MIDDLE_FINGER_SELECTION_RIGHT','MIDDLE_FINGER_CURVED_RIGHT','RING_FINGER_ANGLE_RIGHT','RING_FINGER_SELECTION_RIGHT','RING_FINGER_CURVED_RIGHT','PINKY_ANGLE_RIGHT','PINKY_SELECTION_RIGHT','PINKY_CURVED_RIGHT','PATH_MOVEMENT_RIGHT','APERTURE_MOVEMENT_RIGHT','PATH_ANGLE_MOVEMENT_RIGHT','CHEST_PROXIMITY_LEFT','MOUTH_PROXIMITY_LEFT','NOSE_PROXIMITY_LEFT','EAR_PROXIMITY_LEFT','EYE_PROXIMITY_LEFT','PALM/BACK_PROXIMITY_LEFT','TIPS/WRIST_PROXIMITY_LEFT','URNAL/RADIAL_PROXIMITY_LEFT','THUMB_ANGLE_LEFT','THUMB_SELECTION_LEFT','THUMB_CURVED_LEFT','INDEX_FINGER_ANGLE_LEFT','INDEX_FINGER_SELECTION_LEFT','INDEX_FINGER_CURVED_LEFT','MIDDLE_FINGER_ANGLE_LEFT','MIDDLE_FINGER_SELECTION_LEFT','MIDDLE_FINGER_CURVED_LEFT','RING_FINGER_ANGLE_LEFT','RING_FINGER_SELECTION_LEFT','RING_FINGER_CURVED_LEFT','PINKY_ANGLE_LEFT','PINKY_SELECTION_LEFT','PINKY_CURVED_LEFT','PATH_MOVEMENT_LEFT','APERTURE_MOVEMENT_LEFT','PATH_ANGLE_MOVEMENT_LEFT'])

HAND_JOINTS = ['0 WRIST',
               '1 THUMB_CMC', '2 THUMB_MCP', '3 THUMB_IP', '4 THUMB_TIP',
               '5 INDEX_FINGER_MCP', '6 INDEX_FINGER_PIP', '7 INDEX_FINGER_DIP', '8 INDEX_FINGER_TIP',
               '9 MIDDLE_FINGER_MCP', '10 MIDDLE_FINGER_PIP', '11 MIDDLE_FINGER_DIP', '12 MIDDLE_FINGER_TIP',
               '13 RING_FINGER_MCP', '14 RING_FINGER_PIP', '15 RING_FINGER_DIP', '16 RING_FINGER_TIP',
               '17 PINKY_MCP', '18 PINKY_PIP', '19 PINKY_DIP', '20 PINKY_TIP']

POSE_POINTS = [
    '1 NOSE',
    '2 LEFT_EYE_INNER', '3 LEFT_EYE', '4 LEFT_EYE_OUTER',
    '5 RIGHT_EYE_INNER', '6 RIGHT_EYE', '7 RIGHT_EYE_OUTER',
    '8 LEFT_EAR', '9 RIGHT_EAR',
    '10 MOUTH_LEFT', '11 MOUTH_RIGHT',
    '12 LEFT_SHOULDER', '13 RIGHT_SHOULDER',
    '14 LEFT_ELBOW', '15 RIGHT_ELBOW',
    '16 LEFT_WRIST', '17 RIGHT_WRIST', '18 LEFT_PINKY', '19 RIGHT_PINKY', '20 LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
    'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']


def getCoordinates(p): return np.array([p.x, p.y, p.z])


def getDistance(v1, v2): return np.sqrt(
    np.dot((v1[:-1] - v2[:-1]).T, v1[:-1] - v2[:-1]))


def is_periodic(diffs, value, tolerance=0): return all(
    d-tolerance <= value <= d+tolerance for d in diffs)


def getJointAngle(x):

    a = np.array([x[0][1]['x'], x[0][1]['y'], x[0][1]['z']])
    b = np.array([x[1][1]['x'], x[1][1]['y'], x[1][1]['z']])
    c = np.array([x[2][1]['x'], x[2][1]['y'], x[2][1]['z']])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def getJointAngleNP(a, b, c):

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


class SignLanguagePhonology():
    def __init__(self, hand) -> None:

        self.HAND = hand

        self.INFO_MEMORY = []
        self.CENTER = [np.array([0, 0, 0])]
        self.APERTURE = [np.array([0, 0, 0, 0])]
        self.DIFF = []
        
        self.HAND_MEMORY = []

        self.CURR_START = 0
        self.CURR_STOP = 0
        self.PATH_ANGLE = []
        self.SWITCH = 0
        self.START = []

        # TODO: ADD NON-MANUAL MARKERS
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

    def getJointInfo(self, hand, body_pose, selected):
        """Extracting static information from sign shape including:

        Args:
            results (Landmark): set of lanf

        Returns:
            list: _description_
        """

        marks = []
        logs = []

        if hand:

            center = []

            for joint, point in zip(HAND_JOINTS, list(hand.landmark)):
                marks.append((joint, {
                    'x': point.x,
                    'y': point.y,
                    'z': point.z,
                    'visibility': point.visibility
                }))

                center.append([
                    point.x,
                    point.y,
                    point.z
                ])


            # FIXME: Center by selected finger
            center = np.mean(center, axis=0)

            # Calculates hand movement difference ()
            self.DIFF.append(getDistance(self.CENTER[-1], center))
            self.CENTER.append(center)
            center = np.array(getCoordinates(
                hand.landmark[8]))

            INFO = {
                'LOCATION': self.getLocation(center, body_pose),
                'ORIENTATION': self.getOrientation(marks),
                'FINGER_SELECTION': self.getFingerConfiguration(marks),
            }

            INFO['MOVEMENT'] = self.getMovementFeatures(
                INFO.get('FINGER_SELECTION'))

            for select_ in selected:
                logs.append(INFO[select_])

            self.INFO_MEMORY.append(INFO)

            return logs

        else:
            center = np.array([0, 0, 0])
            self.DIFF.append(getDistance(self.CENTER[-1], center))
            self.CENTER.append(center)

            self.APERTURE.append(np.array([0, 0, 0, 0]))

            self.SWITCH = 0
            self.START.append(self.SWITCH)

            return None

    def getFingerConfiguration(self, marks):

        # TODO: ADD CONTACT FEATURE #logs.append(('Thumb-Index Tips:',  (marks[4][1]['x'] - marks[8][1]['x'],marks[4][1]['y'] - marks[8][1]['y'])   ))

        logs = []

        THUMB_0 = getJointAngle(marks[2:5])
        logs.append({
            'FINGER': 'THUMB',
            'ANGLE_0': THUMB_0,
            'is_selected': bool(THUMB_0 >= 160),
            'is_curved': bool(THUMB_0 <= 160 and THUMB_0 >= 90),
        })

        INDEX_0 = getJointAngle(marks[5:9])
        INDEX_1 = getJointAngle([marks[0], marks[6], marks[8]])
        logs.append({
            'finger': 'INDEX_FINGER',
            'ANGLE_0': INDEX_0,
            'ANGLE_1': INDEX_1,
            'is_selected': bool(INDEX_1 >= 90),
            'is_curved': bool(INDEX_1 <= 160 and INDEX_1 >= 90),
        })

        MIDDLE_0 = getJointAngle(marks[9:13])
        MIDDLE_1 = getJointAngle([marks[0], marks[9], marks[12]])
        logs.append({
            'finger': 'MIDDLE_FINGER',
            'ANGLE_0': MIDDLE_0,
            'ANGLE_1': MIDDLE_1,
            'is_selected': bool(MIDDLE_1 >= 90),
            'is_curved': bool(MIDDLE_1 <= 165 and MIDDLE_1 >= 90),
        })

        RING_0 = getJointAngle(marks[13:17])
        RING_1 = getJointAngle([marks[0], marks[13], marks[16]])
        logs.append({
            'finger': 'RING_FINGER',
            'ANGLE_0': RING_0,
            'ANGLE_1': RING_1,
            'is_selected': bool(RING_1 >= 90),
            'is_curved': bool(RING_1 <= 165 and RING_1 >= 90),
        })

        PINKY_0 = getJointAngle(marks[17:])
        PINKY_1 = getJointAngle([marks[0], marks[17], marks[-1]])
        logs.append({
            'finger': 'PINKY',
            'ANGLE_0': PINKY_0,
            'ANGLE_1': PINKY_1,
            'is_selected': bool(PINKY_1 >= 90),
            'is_curved': bool(PINKY_1 <= 165 and PINKY_1 >= 90),
        })

        self.APERTURE.append(np.array([INDEX_1, MIDDLE_1, RING_1, PINKY_1]))

        return logs

    def getOrientation(self, marks):

        orientation = []

        if (marks[4][1]['x'] - marks[17][1]['x']) + (marks[4][1]['y'] - marks[17][1]['y']) > 0.00:
            orientation.append(
                ('palm' if self.HAND == 'right' else 'back', marks[4][1]['x'] - marks[17][1]['x'], marks[4][1]['y'] - marks[17][1]['y']))
        else:
            orientation.append(
                ('back' if self.HAND == 'right' else 'palm', marks[4][1]['x'] - marks[17][1]['x'], marks[4][1]['y'] - marks[17][1]['y']))

        # FIXME: Error at wrist (z axis problem)
        if marks[0][1]['z'] > marks[9][1]['z']:
            orientation.append(('tips', marks[0][1]['z'] - marks[9][1]['z']))
        else:
            orientation.append(('wrist', marks[0][1]['z'] - marks[9][1]['z']))

        if marks[2][1]['z'] > marks[17][1]['z']:
            orientation.append(('urnal', marks[2][1]['z'] - marks[17][1]['z']))
        else:
            orientation.append(
                ('radial', marks[2][1]['z'] - marks[17][1]['z']))

        return {'ORIENTATION': orientation}

    def getLocation(self, center, body_pose):

        # TODO: ADD MAJOR/MINOR LOCATION

        # FIXME: CHANGE TO MEAN(POINTS_X)
        body_poses = {
            # LEFT
            'EYE': (getCoordinates(body_pose.landmark[3]), getDistance(center, getCoordinates(body_pose.landmark[3]))),
            # LEFT
            'EAR': (getCoordinates(body_pose.landmark[7]), getDistance(center, getCoordinates(body_pose.landmark[7]))),
            # LEFT
            'NOSE': (getCoordinates(body_pose.landmark[0]), getDistance(center, getCoordinates(body_pose.landmark[0]))),
            # LEFT
            'MOUTH': (getCoordinates(body_pose.landmark[9]), getDistance(center, getCoordinates(body_pose.landmark[9]))),
            # LEFT
            'CHEST': (getCoordinates(body_pose.landmark[11]), getDistance(center, getCoordinates(body_pose.landmark[11]))),
        }

        return dict(sorted(body_poses.items(), key=lambda x: x[1][1]))

    def getMovementFeatures(self, marks):

        logs = []

        # FIXME: SIGN BOUNDARY -> TRAINABLE PARAMETER
        # TODO: SIGN BOUNDARY
        # - BY SELECT FINGER CHANGE ( SELECTED FINGER CONSTRAINT )
        # - BY MAJOR LOCATION CHANGE ( MAJOR LOCATION CONSTRAINT )
        if np.mean(self.DIFF[-3:]) > 0.007:
            self.SWITCH = 1
            self.CURR_START += 1
            self.CURR_STOP = 0

            # Direction
            logs.append(
                {'PATH': 'high' if self.CENTER[-2][1] > self.CENTER[-1][1] else 'low'})

            # Aperture
            logs.append({'APERTURE': 'opening' if sum(
                self.APERTURE[-1] - self.APERTURE[-2]) > 0 else 'closing'})

            SELECTED = self.CENTER[-self.CURR_START:]

            if len(SELECTED) > 3:
                self.PATH_ANGLE.append(getJointAngleNP(
                    np.array(SELECTED[0])[:2], np.array(SELECTED[round(len(SELECTED) / 2)])[:2], np.array(SELECTED[-1])[:2]))

                MPA = np.mean(self.PATH_ANGLE)
                logs.append(
                    {'PATH_ANGLE': ('curve' if MPA < 90 else 'straight', MPA)})

        else:
            self.CURR_START = 0
            self.CURR_STOP += 1
            self.SWITCH = 0
            self.PATH_ANGLE = []

        self.START.append(self.SWITCH)

        return logs

    def getMovementInformation(self, view_window, window_size, metrics):

        INFO = {}

        if 'Center_Diff' in metrics:
            INFO['Center_Diff'] = self.DIFF[-view_window:]

        if 'Center_MovAVG_x' in metrics:
            INFO['Center_MovAVG_x'] = pd.Series(np.array(
                self.CENTER[-view_window:])[:, 0]).rolling(window_size).mean().tolist()
        if 'Center_MovAVG_y' in metrics:
            INFO['Center_MovAVG_y'] = pd.Series(np.array(
                self.CENTER[-view_window:])[:, 1]).rolling(window_size).mean().tolist()
        if 'Center_MovAVG_z' in metrics:
            INFO['Center_MovAVG_z'] = pd.Series(np.array(
                self.CENTER[-view_window:])[:, 2]).rolling(window_size).mean().tolist()
        if 'Aperture_MovAVG_0' in metrics:
            INFO['Aperture_MovAVG_0'] = pd.Series(np.array(
                self.APERTURE[-view_window:])[:, 0] / 180).rolling(window_size).mean().tolist()
        if 'Aperture_MovAVG_1' in metrics:
            INFO['Aperture_MovAVG_1'] = pd.Series(np.array(
                self.APERTURE[-view_window:])[:, 1] / 180).rolling(window_size).mean().tolist()
        if 'Aperture_MovAVG_2' in metrics:
            INFO['Aperture_MovAVG_2'] = pd.Series(np.array(
                self.APERTURE[-view_window:])[:, 2] / 180).rolling(window_size).mean().tolist()
        if 'Aperture_MovAVG_3' in metrics:
            INFO['Aperture_MovAVG_3'] = pd.Series(np.array(
                self.APERTURE[-view_window:])[:, 3] / 180).rolling(window_size).mean().tolist()

        if 'SWITCH' in metrics:
            INFO['SWITCH'] = self.START[-view_window:]

        min_len = min([len(i) for i in INFO.values()])

        if 'LSE_x' in metrics:
            LSE_x, LSE_y = self.getArchError(
                INFO['Center_MovAVG_x'], INFO['Center_MovAVG_y'], min_len)
            INFO['LSE_x'] = LSE_x
            INFO['LSE_y'] = LSE_y

        # return pd.DataFrame(INFO)
        return pd.DataFrame({label: arr[:min_len] for label, arr in INFO.items()})

    def getArchError(self, x, y, min_len):

        x = pd.Series(x).replace(np.inf, np.nan).replace(-np.inf,
                                                         np.nan).dropna().to_numpy()
        y = pd.Series(y).replace(np.inf, np.nan).replace(-np.inf,
                                                         np.nan).dropna().to_numpy()

        lines_x = np.zeros(x.shape[0])
        lines_y = np.zeros(x.shape[0])

        if x.shape[0]:
            A = np.vstack([np.array(list(range(len(x)))), np.ones(len(x))]).T
            (m_x, c_x), residuals, RANK, sing = np.linalg.lstsq(A, x, rcond=None)
            # lines_x[np.array(self.START[-x.shape[0]:]) == 1] = m_x*np.array(list(range( lines_x[np.array(self.START[-x.shape[0]:]) == 1].shape[0] ))) + c_x
            lines_x[np.array(self.START[-x.shape[0]:]) == 1] = (
                (np.mean(residuals) / np.array(self.START[-x.shape[0]:]).sum()) * 10)

        if y.shape[0]:
            A = np.vstack([np.array(list(range(len(y)))), np.ones(len(y))]).T
            (m_y, c_y), residuals, RANK, sing = np.linalg.lstsq(A, y, rcond=None)
            # lines_y[np.array(self.START[-x.shape[0]:]) == 1] = m_y*np.array(list(range( lines_y[np.array(self.START[-x.shape[0]:]) == 1].shape[0] ))) + c_y
            lines_y[np.array(self.START[-x.shape[0]:]) == 1] = (
                (np.mean(residuals) / np.array(self.START[-x.shape[0]:]).sum()) * 10)

        if x.shape[0] < min_len:
            lines_x = np.append(lines_x, np.zeros(min_len-x.shape[0]))
            lines_y = np.append(lines_y, np.zeros(min_len-x.shape[0]))

        return lines_x, lines_y

class TwoHandedPhonology():

    def __init__(self) -> None:
        # TODO: ADD HANDEDNESS
        # TODO: ADD TWO-HANDED CORRELATION (SYMMETRY / ASSYMETRY / CONTACT)

        # self.CENTER

        pass

    def findTemporalAlignment(self,
                              left: SignLanguagePhonology,
                              right: SignLanguagePhonology,
                              view_window: int,
                              window_size: int,
                              metrics: list):

        INFO = {}

        if 'Center_MovAVG_Left' in metrics:
            INFO['Center_MovAVG_Left'] = pd.Series(np.mean(
                left.CENTER[-view_window:], axis=1) / 180).rolling(window_size).mean().tolist()
        if 'Center_MovAVG_Right' in metrics:
            INFO['Center_MovAVG_Right'] = pd.Series(np.mean(
                right.CENTER[-view_window:], axis=1) / 180).rolling(window_size).mean().tolist()
        if 'Aperture_MovAVG_Left' in metrics:
            INFO['Aperture_MovAVG_Left'] = pd.Series(np.mean(
                left.APERTURE[-view_window:], axis=1) / 180).rolling(window_size).mean().tolist()
        if 'Aperture_MovAVG_Right' in metrics:
            INFO['Aperture_MovAVG_Right'] = pd.Series(np.mean(
                right.APERTURE[-view_window:], axis=1) / 180).rolling(window_size).mean().tolist()
        if 'SWITCH_Left' in metrics:
            INFO['SWITCH_Left'] = left.START[-view_window:]
        if 'SWITCH_Right' in metrics:
            INFO['SWITCH_Right'] = right.START[-view_window:]

        min_len = min([len(i) for i in INFO.values()])

        return pd.DataFrame({label: arr[:min_len] for label, arr in INFO.items()})

class LexicalClassification():
    
    def __init__(self):
        
        # TODO: EXPERIMENTS
        # - POSE + DTW
        # - PRIOR_FEATS + DTW
        # - PROPOSED_FEATS + DTW
        
        # EXPERIMENTS: DTW TYPES
        # - WEIGHTED DTW
                
        self.LEXICON = []
        self.VECTORS = []
        
        self.PREDICTED = []
        
        self.importLexicon(INFO_TYPE='feat', METRICS=None)
        
    def importLexicon(self, INFO_TYPE: str, METRICS: list[str]):
        
        file = os.listdir('data/vectors/feats/')[-1]
                
        ITEMS = pickle.loads(open('data/vectors/feats/'+file, 'rb').read())
        
        for label, INFO in tqdm(ITEMS.items()):
            
            self.LEXICON.append(label)
            self.VECTORS.append(self.processVectors(INFO))

    def processVectors(self, INFO):
       
       # SET INFORMATION:
       # - LOCATION: Signal to MAJOR LOCATIONS ( MAJOR LOCATION CONSTRAINT)
       # - ORIENTATION: Signal to 6 ORIENTATION (DIFF RELATIVE SCORE TO 3 PAIRS)
       # - FINGER_SELECTION: Signal to 5 FINGER APERTURES (TODO: SHOULD ADD SELECTION/CURVE FEATS) ( SELECTED FINGER CONSTRAINT)
       # - MOVEMENT: Signal to 2 MOVEMENT FEATS ( PATH: (0,1), APERATURE: (0,1) )
       
       
       VECTORS = self.generateVector(INFO['right_log'], 'right')
       VECTORS.update(self.generateVector(INFO['left_log'], 'left'))
       
       return VECTORS
       
    def generateVector(self, points, hand):
        
        FEAT_DICT = defaultdict(list)
        
        for point in points:
        
            for location, (_, proximity) in point['LOCATION'].items():
                
                FEAT_DICT[f"{location}_PROXIMITY_{hand.upper()}"].append(proximity)

            # FIXME: UNNECESSARY DICT
            for orient,rotation in zip(['PALM/BACK', 'TIPS/WRIST', 'URNAL/RADIAL' ],point['ORIENTATION']['ORIENTATION']):
                
                FEAT_DICT[f"{orient}_PROXIMITY_{hand.upper()}"].append(list(rotation)[-1])   
            
            for hand_dict in point['FINGER_SELECTION']:
                
                ff = 'FINGER' if 'FINGER' in hand_dict.keys() else 'finger'                 
                
                FEAT_DICT[f"{hand_dict[ff]}_ANGLE_{hand.upper()}"].append(hand_dict['ANGLE_1' if hand_dict[ff] != 'THUMB' else 'ANGLE_0'])
            
                FEAT_DICT[f"{hand_dict[ff]}_SELECTION_{hand.upper()}"].append(1 if hand_dict['is_selected'] else 0)
                
                FEAT_DICT[f"{hand_dict[ff]}_CURVED_{hand.upper()}"].append(1 if hand_dict['is_curved'] else 0)

            for feats in point['MOVEMENT']:
                            
                feat_name, val = list(feats.items())[0]
                
                FEAT_DICT[f"{feat_name}_MOVEMENT_{hand.upper()}"].append(val[-1])
                
        
        FEAT_DICT = {
            FEATURE: pd.Series(VALS)  for FEATURE, VALS in FEAT_DICT.items()
        }
        
        return FEAT_DICT
    
    def predict(self, INFO):
        
        SIMILARITY = {}

        BASE = self.processVectors(INFO)

        zip_lex = list(zip(self.LEXICON,self.VECTORS))

        for word, candidate in tqdm(zip_lex):

            KEYS = list(KEY_SET.intersection(set(list(candidate.keys())).intersection(BASE.keys())))
            # v0
            sim = 0
            kk = 0
            for dim in KEYS:
                #sim += dtw(BASE[dim], candidate[dim])
                try:
                    v1, v2 = np.array([BASE[dim].to_list()]), np.array([candidate[dim].to_list()])
                    min_len = min([v1.shape[1],v2.shape[1]])
                    
                    #pad = math.floor(abs(v1.shape[1]-v2.shape[1]) / 2)

                    sim += fastdtw(v1[:,:min_len], v2[:,:min_len])[0]
                    kk += 1
                except:
                    pass


            if kk: SIMILARITY[word] = sim / kk
        
        vla = sorted(SIMILARITY.items(), key=lambda x:x[1], reverse=False)
        if vla:
            self.PREDICTED.append(vla[0])
                
        
                
        
    
        
        
        