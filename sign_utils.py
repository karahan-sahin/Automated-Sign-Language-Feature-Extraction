import pandas as pd
import numpy as np
from scipy.spatial import distance

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


class SignLanguagePhonology():
    def __init__(self) -> None:

        self.CENTER = [np.array([0, 0, 0])]
        self.APERTURE = [np.array([0, 0, 0, 0])]
        self.DIFF = []

        self.CURR = 0
        self.SWITCH = 0
        self.START = []

        # TODO: ADD HANDEDNESS
        # TODO: ADD TWO-HANDED CORRELATION (SYMMETRY / ASSYMETRY / CONTACT)
        # Face
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

            return logs

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
                ('palm', marks[4][1]['x'] - marks[17][1]['x'], marks[4][1]['y'] - marks[17][1]['y']))
        else:
            orientation.append(
                ('back', marks[4][1]['x'] - marks[17][1]['x'], marks[4][1]['y'] - marks[17][1]['y']))

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
        if np.mean(self.DIFF[-3:]) > 0.007:
            self.SWITCH = 1
            self.CURR += 1

            # Direction
            logs.append(
                {'PATH': 'high' if self.CENTER[-2][1] > self.CENTER[-1][1] else 'low'})

            # Aperture
            logs.append({'APERTURE': 'opening' if sum(
                self.APERTURE[-1] - self.APERTURE[-2]) > 0 else 'closing'})

        else:
            self.CURR = 0
            self.SWITCH = 0

        self.START.append(self.SWITCH)

        return logs

    def getMovementInformation(self, view_window, window_size, metrics):

        Center_Diff = self.DIFF[-view_window:]

        Center_MovAVG_x = pd.Series(np.array(
            self.CENTER[-view_window:])[:, 0]).rolling(window_size).mean().tolist()
        Center_MovAVG_y = pd.Series(np.array(
            self.CENTER[-view_window:])[:, 1]).rolling(window_size).mean().tolist()
        Center_MovAVG_z = pd.Series(np.array(
            self.CENTER[-view_window:])[:, 2]).rolling(window_size).mean().tolist()
        
        
        Aperture_MovAVG_0 = pd.Series(np.array(
            self.APERTURE[-view_window:])[:, 0] / 180 ).rolling(window_size).mean().tolist()
        Aperture_MovAVG_1 = pd.Series(np.array(
            self.APERTURE[-view_window:])[:, 1] / 180 ).rolling(window_size).mean().tolist()
        Aperture_MovAVG_2 = pd.Series(np.array(
            self.APERTURE[-view_window:])[:, 2] / 180 ).rolling(window_size).mean().tolist()
        Aperture_MovAVG_3 = pd.Series(np.array(
            self.APERTURE[-view_window:])[:, 3] / 180 ).rolling(window_size).mean().tolist()

        SWITCH = self.START[-view_window:]

        min_len = min(
            len(Center_Diff),
            len(Center_MovAVG_x),
            len(Aperture_MovAVG_0),
            len(SWITCH))

        INFO = {}

        if 'Center_Diff' in metrics: INFO['Center_Diff'] = Center_Diff[:min_len]
        if 'Center_MovAVG_x' in metrics: INFO['Center_MovAVG_x'] = Center_MovAVG_x[:min_len]
        if 'Center_MovAVG_y' in metrics: INFO['Center_MovAVG_y'] = Center_MovAVG_y[:min_len]
        if 'Center_MovAVG_z' in metrics: INFO['Center_MovAVG_z'] = Center_MovAVG_z[:min_len]
        if 'INDEX_FINGER' in metrics: INFO['INDEX_FINGER'] = Aperture_MovAVG_0[:min_len]
        if 'MIDDLE_FINGER' in metrics: INFO['MIDDLE_FINGER'] = Aperture_MovAVG_1[:min_len]
        if 'RING_FINGER' in metrics: INFO['RING_FINGER'] = Aperture_MovAVG_2[:min_len]
        if 'PINKY' in metrics: INFO['PINKY'] = Aperture_MovAVG_3[:min_len]
        if 'SWITCH' in metrics: INFO['SWITCH'] = SWITCH[:min_len]

        return pd.DataFrame(INFO)
