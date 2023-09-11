import pickle
from tqdm import tqdm
import os
from collections import defaultdict
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
        
        self.importLexicon(INFO_TYPE='feat', METRICS=None)
        
    def importLexicon(self, INFO_TYPE: str, METRICS: list[str]):
        
        file = os.listdir('data/vectors/feats/')[-1]
                
        ITEMS = pickle.loads(open('data/vectors/feats/'+file, 'rb').read())
        
        for label, INFO in tqdm(ITEMS.items()):
            
            self.LEXICON.append(label)
            self.VECTORS.append(self.processVectors(INFO, INFO_TYPE, METRICS))

    def processVectors(self, INFO, INFO_TYPE, METRICS):
       
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

CLS = LexicalClassification()