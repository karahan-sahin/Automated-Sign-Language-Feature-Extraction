HAND_JOINTS = [
    'WRIST',
    'THUMB_CMC',
    'THUMB_MCP',
    'THUMB_IP',
    'THUMB_TIP',
    'INDEX_FINGER_MCP',
    'INDEX_FINGER_PIP',
    'INDEX_FINGER_DIP',
    'INDEX_FINGER_TIP',
    'MIDDLE_FINGER_MCP',
    'MIDDLE_FINGER_PIP',
    'MIDDLE_FINGER_DIP',
    'MIDDLE_FINGER_TIP',
    'RING_FINGER_MCP',
    'RING_FINGER_PIP',
    'RING_FINGER_DIP',
    'RING_FINGER_TIP',
    'PINKY_MCP',
    'PINKY_PIP',
    'PINKY_DIP',
    'PINKY_TIP'
]

POSE_POINTS = [
    'NOSE',
    'LEFT_EYE_INNER',
    'LEFT_EYE',
    'LEFT_EYE_OUTER',
    'RIGHT_EYE_INNER',
    'RIGHT_EYE',
    'RIGHT_EYE_OUTER',
    'LEFT_EAR',
    'RIGHT_EAR',
    'MOUTH_LEFT',
    'MOUTH_RIGHT',
    'LEFT_SHOULDER',
    'RIGHT_SHOULDER',
    'LEFT_ELBOW',
    'RIGHT_ELBOW',
    'LEFT_WRIST',
    'RIGHT_WRIST',
    'LEFT_PINKY',
    'RIGHT_PINKY',
    'LEFT_INDEX',
    'RIGHT_INDEX',
    'LEFT_THUMB',
    'RIGHT_THUMB',
    'LEFT_HIP',
    'RIGHT_HIP',
    'LEFT_KNEE',
    'RIGHT_KNEE',
    'LEFT_ANKLE',
    'RIGHT_ANKLE',
    'LEFT_HEEL',
    'RIGHT_HEEL',
    'LEFT_FOOT_INDEX',
    'RIGHT_FOOT_INDEX'
]

COLS = [
    'RIGHT_THUMB_ANGLE_DIP',
    'RIGHT_TIPS/WRIST_SCORE',
    'LEFT_PINKY_ANGLE_DIP_CURVE',
    'LEFT_SETTING_SCORE',
    'LEFT_THUMB_ANGLE_DIP_SELECT',
    'LEFT_RADIAL/URNAL_LABEL',
    'LEFT_TIPS/WRIST_LABEL',
    'RIGHT_INDEX_FINGER_ANGLE_DIP_SELECT',
    'RIGHT_RADIAL/URNAL_SCORE',
    'LEFT_PINKY_ANGLE_DIP',
    'RIGHT_CHEST',
    'RIGHT_MOUTH',
    'LEFT_ARM',
    'LEFT_MOUTH',
    'LEFT_TIPS/WRIST_SCORE',
    'RIGHT_PINKY_ANGLE_DIP_CURVE',
    'LEFT_INDEX_FINGER_ANGLE_DIP_CURVE',
    'LEFT_PINKY_ANGLE_DIP_SELECT',
    'LEFT_INDEX_FINGER_ANGLE_DIP',
    'LEFT_REPETITION_SCORE',
    'RIGHT_MIDDLE_FINGER_ANGLE_DIP_SELECT',
    'RIGHT_MIDDLE_FINGER_ANGLE_DIP_CURVE',
    'RIGHT_PINKY_ANGLE_DIP_SELECT',
    'RIGHT_PINKY_ANGLE_DIP',
    'LEFT_REPETITION_LABEL',
    'LEFT_SETTING_LABEL',
    'RIGHT_THUMB_ANGLE_DIP_SELECT',
    'RIGHT_EAR',
    'LEFT_RING_FINGER_ANGLE_DIP_CURVE',
    'LEFT_NOSE',
    'RIGHT_INDEX_FINGER_ANGLE_DIP_CURVE',
    'LEFT_RING_FINGER_ANGLE_DIP',
    'LEFT_SHOULDER',
    'IN_FRONT_SCORE',
    'RIGHT_TIPS/WRIST_LABEL',
    'RIGHT_RADIAL/URNAL_LABEL',
    'LEFT_RADIAL/URNAL_SCORE',
    'LEFT_EYE',
    'LEFT_PALM/BACK_SCORE',
    'ABOVE/BELOW_LABEL',
    'RIGHT_PALM/BACK_LABEL',
    'RIGHT_NOSE',
    'LEFT_PALM/BACK_LABEL',
    'LEFT_MIDDLE_FINGER_ANGLE_DIP_SELECT',
    'LEFT_INDEX_FINGER_ANGLE_DIP_SELECT',
    'RIGHT_SHOULDER',
    'RIGHT_RING_FINGER_ANGLE_DIP_CURVE',
    'RIGHT_INDEX_FINGER_ANGLE_DIP',
    'LEFT_THUMB_ANGLE_DIP',
    'LEFT_MIDDLE_FINGER_ANGLE_DIP_CURVE',
    'RIGHT_HIP',
    'RIGHT_RING_FINGER_ANGLE_DIP',
    'RIGHT_RING_FINGER_ANGLE_DIP_SELECT',
    'IN_FRONT_LABEL',
    'RIGHT_EYE',
    'LEFT_HIP',
    'RIGHT_PALM/BACK_SCORE',
    'RIGHT_SETTING_LABEL',
    'LEFT_THUMB_ANGLE_DIP_CURVE',
    'RIGHT_CURVE_LABEL',
    'ABOVE/BELOW_SCORE',
    'LEFT_RING_FINGER_ANGLE_DIP_SELECT',
    'RIGHT_MIDDLE_FINGER_ANGLE_DIP',
    'LEFT_CURVE_LABEL',
    'LEFT_EAR',
    'RIGHT_SETTING_SCORE',
    'LEFT_CHEST',
    'RIGHT_ARM',
    'LEFT_MIDDLE_FINGER_ANGLE_DIP',
    'RIGHT_THUMB_ANGLE_DIP_CURVE'
]

MODEL_DIR = {
    'baseline': 'lib/models/baseline',
    'LSTM Auto-encoder': 'lib/models/lstm_autoencoder',
}


HAND_IDX2POINT = dict(enumerate(HAND_JOINTS))
HAND_POINT2IDX = { point: idx for idx,
 point in HAND_IDX2POINT.items() }

BODY_IDX2POINT = dict(enumerate(POSE_POINTS))
BODY_POINT2IDX = { point: idx for idx,
 point in BODY_IDX2POINT.items() }

COL_IDX2FEAT = dict(enumerate(COLS))
COL_FEAT2IDX = { FEAT: idx for idx,
 FEAT in COL_IDX2FEAT.items() }