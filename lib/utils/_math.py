import numpy as np
from scipy.spatial import distance
from ..phonology.sign_utils import BODY_POINT2IDX, HAND_POINT2IDX

def getCoordinates(p, z=True): 
    """Get coordinates from mediapipe landmarks

    Args:
        p (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.array([ p.x, p.y ] + [p.z] if z else [] )

def getDistance(v1: np.array, v2: np.array): 
    """Get cosine similarity between points

    Args:
        v1 (np.array): _description_
        v2 (np.array): _description_

    Returns:
        _type_: _description_
    """
    
    return np.sqrt( np.dot((v1 - v2).T, v1 - v2))

def getCosineAngle(a, b, c):

    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)
