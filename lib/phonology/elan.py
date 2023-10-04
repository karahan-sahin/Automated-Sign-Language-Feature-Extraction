import cv2
import numpy as np
import pandas as pd
from pympi.Elan import Eaf
from .sign_utils import FEAT_TIER_MAP, NUMERICAL

class ELANWriter():

    def __init__(self, feature_df, video_path, percentage, roll_window, threshold) -> None:

        self.FEATURE_DF = feature_df
        self.processDf()
        self.video_file_path = video_path

        self.ELAN = Eaf(author='pympi')
        self.createFile(['FINGER_COORDINATION', 'MANNER', 'ORIENTATION', 'LOCATION'])

        self.VIDEO_LENGTH = None
        self.VIDEO_FPS = None
        
        self.getVideoProperties()
        self.parseSegments(percentage, roll_window, threshold)

    def processDf(self):
        
        # Select Major Max Location
        self.FEATURE_DF['RIGHT_LOCATION'] = self.FEATURE_DF[['RIGHT_HIP',  'RIGHT_CHEST',  'RIGHT_ARM', 'RIGHT_SHOULDER',  'RIGHT_EAR',  'RIGHT_MOUTH',  'RIGHT_EYE',  'RIGHT_NOSE' ]].apply(lambda x: x.sort_values().index[0], axis=1).to_list()
        self.FEATURE_DF['LEFT_LOCATION'] = self.FEATURE_DF[['LEFT_HIP',  'LEFT_CHEST',  'LEFT_ARM', 'LEFT_SHOULDER',  'LEFT_EAR',  'LEFT_MOUTH',  'LEFT_EYE',  'LEFT_NOSE' ]].apply(lambda x: x.sort_values().index[0], axis=1).to_list()
        
        # Select Finger Conf. / Selected
        self.handleFingerSelection( FTYPE = 'SELECT', HAND = 'RIGHT', JTYPE = 'PIP' )
        self.handleFingerSelection( FTYPE = 'SELECT', HAND = 'LEFT', JTYPE = 'PIP' )
        self.handleFingerSelection( FTYPE = 'CURVE', HAND = 'LEFT', JTYPE = 'PIP' )
        self.handleFingerSelection( FTYPE = 'CURVE', HAND = 'RIGHT', JTYPE = 'PIP' )

        
    def handleFingerSelection(self,
                              FTYPE = 'SELECT',
                              HAND = 'RIGHT',
                              JTYPE = 'PIP'):
        
        self.FEATURE_DF[f'{HAND}_{FTYPE}'] = self.FEATURE_DF[[
            f'{HAND}_THUMB_ANGLE_{JTYPE}_{FTYPE}',
            f'{HAND}_INDEX_FINGER_ANGLE_{JTYPE}_{FTYPE}',
            f'{HAND}_MIDDLE_FINGER_ANGLE_{JTYPE}_{FTYPE}',
            f'{HAND}_RING_FINGER_ANGLE_{JTYPE}_{FTYPE}',
            f'{HAND}_PINKY_ANGLE_{JTYPE}_{FTYPE}'
        ]].apply(lambda x: self.selectedFingers(x), axis=1)
        

    def selectedFingers(self, row: pd.Series): return 'all' if row.all() else '+'.join(list(map(lambda x: str(self.matchIdx(x)),row[row].index)))

    def matchIdx(self, nn): 
        JOINT_IDX = { 'THUMB': 1, 'INDEX': 2, 'MIDDLE': 3, 'RING': 4, 'PINKY': 5 }
        for name, idx in JOINT_IDX.items():
            if name in nn : return idx 


    def extractFeatures(self, strategy='basic', output_file='Untitled', params=dict | None):

        if strategy == 'basic':
            self.extractBasic(params)
            
        self.ELAN.to_file(output_file+'.eaf')

    def extractBasic(self, params=None):

        print(' ** Extracting features ** ')
        
        for col, tier_name in FEAT_TIER_MAP.items():
            seg_start = 0
            seg_label = None
            
            if not col in self.FEATURE_DF.columns:
                print('Feature cannot be written', col)
                continue
            
            else:
                for idx, label in list(enumerate(self.FEATURE_DF[col].replace(pd.NA, None).to_list())):
                    idx = int((idx / self.VIDEO_FPS) * 1000)
                    if seg_label != label:
                        if seg_label != None and seg_label != '0':
                            # try:
                            self.ELAN.add_annotation(
                                tier_name,
                                seg_start,
                                idx-1,
                                value=seg_label,
                                svg_ref=None)
                            # except ValueError:
                            #     print('Cannot write annotation:',
                            #         seg_label, seg_start, idx-1)

                        seg_start = idx
                        seg_label = str(label)
                
                try:
                    self.ELAN.add_annotation(
                            tier_name,
                            seg_start,
                            idx-1,
                            value=seg_label,
                            svg_ref=None) 
                except: pass
                

    def getVideoProperties(self):

        cap = cv2.VideoCapture(self.video_file_path)
        self.VIDEO_LENGTH = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS)

    def showTierTree(self):
        def print_tree(tier_name, indent='|-'):
            children = self.ELAN.get_child_tiers_for(tier_name)
            if children:
                for tt in children:
                    print(indent, tt)
                    print_tree(tt, indent='|----'+indent)

        print('MANUAL')
        print_tree('MANUAL')

    def createFile(self,
                   selected=['FINGER_COORDINATION', 'MANNER', 'ORIENTATION', 'LOCATION']):

        self.ELAN.add_tier(
            'GLOSS-SEGMENTS',
            ling='default-lt'
        )

        self.ELAN.add_tier(
            'MANUAL',
            ling='default-lt'
        )

        for HAND in ['LEFT', 'RIGHT']:

            ##################

            if 'FINGER_COORDINATION' in selected:

                self.ELAN.add_tier(
                    'FINGER_COORDINATION', ling='default-lt', parent='MANUAL')

                self.ELAN.add_tier(
                    f'{HAND}_FINGER_COORDINATION', ling='default-lt', parent='FINGER_COORDINATION')

                self.ELAN.add_tier(
                    f'{HAND}_FINGER_COORDINATION_SELECTION', ling='default-lt', parent=f'{HAND}_FINGER_COORDINATION')

                self.ELAN.add_tier(
                    f'{HAND}_FINGER_COORDINATION_CURVE', ling='default-lt', parent=f'{HAND}_FINGER_COORDINATION')

                self.ELAN.add_tier(
                    f'{HAND}_FINGER_COORDINATION_OPEN/CLOSE', ling='default-lt', parent=f'{HAND}_FINGER_COORDINATION')

            ###########################

            if 'MANNER' in selected:

                self.ELAN.add_tier(
                    'MANNER', ling='default-lt', parent='MANUAL')

                self.ELAN.add_tier(
                    'MANNER_PATH', ling='default-lt', parent='MANNER')

                self.ELAN.add_tier(
                    'MANNER_TEMPORAL', ling='default-lt', parent='MANNER')

                self.ELAN.add_tier(
                    'MANNER_TEMPORAL_REPETITION', ling='default-lt', parent='MANNER_TEMPORAL')

                self.ELAN.add_tier(
                    'MANNER_TEMPORAL_ALTERNATION', ling='default-lt', parent='MANNER_TEMPORAL')

                self.ELAN.add_tier(
                    'MANNER_TEMPORAL_SYNC/ASYNC', ling='default-lt', parent='MANNER_TEMPORAL')

                self.ELAN.add_tier(
                    'MANNER_TEMPORAL_CONTACT', ling='default-lt', parent='MANNER_TEMPORAL')

                self.ELAN.add_tier(
                    'MANNER_SPACIAL', ling='default-lt', parent='MANNER')

                self.ELAN.add_tier(
                    'MANNER_SPACIAL_ABOVE', ling='default-lt', parent='MANNER_SPACIAL')

                self.ELAN.add_tier(
                    'MANNER_SPACIAL_IN_FRONT_OF', ling='default-lt', parent='MANNER_SPACIAL')

                self.ELAN.add_tier(
                    'MANNER_SPACIAL_INTERWOVEN', ling='default-lt', parent='MANNER_SPACIAL')

            ###########################

            if 'ORIENTATION' in selected:

                self.ELAN.add_tier(
                    'ORIENTATION', ling='default-lt', parent='MANUAL')

                self.ELAN.add_tier(
                    f'{HAND}_ORIENTATION', ling='default-lt', parent='ORIENTATION')

                self.ELAN.add_tier(
                    f'{HAND}_ORIENTATION_PALM/BACK', ling='default-lt', parent=f'{HAND}_ORIENTATION')

                self.ELAN.add_tier(
                    f'{HAND}_ORIENTATION_TIPS/WRIST', ling='default-lt', parent=f'{HAND}_ORIENTATION')

                self.ELAN.add_tier(
                    f'{HAND}_ORIENTATION_RADIAL/URNAL', ling='default-lt', parent=f'{HAND}_ORIENTATION')

            ###########################

            if 'LOCATION' in selected:

                self.ELAN.add_tier(
                    'LOCATION', ling='default-lt', parent='MANUAL')

                self.ELAN.add_tier(
                    f'{HAND}_LOCATION', ling='default-lt', parent='LOCATION')

                self.ELAN.add_tier(
                    f'{HAND}_LOCATION_MAJOR', ling='default-lt', parent=f'{HAND}_LOCATION')

                self.ELAN.add_tier(
                    f'{HAND}_LOCATION_SETTING', ling='default-lt', parent=f'{HAND}_LOCATION')

            ###########################

        self.ELAN.add_linked_file(
            file_path=self.video_file_path,
            mimetype='mp4'
        )

    def parseSegments(self, percentage='75%', roll=10, threshold = 0.3):

        pltt = self.FEATURE_DF[NUMERICAL]

        pltt = pltt.replace(np.nan, 0)

        mov = pd.Series(np.zeros(pltt.shape[0]))

        for col in NUMERICAL:
            mov+= (pltt[col].rolling(roll).mean()) 
            
        pltt['mov'] = mov 
        diff = []
        for idx, dd in enumerate(pltt['mov'].to_list()[:-1]):
            diff.append(abs(dd - pltt['mov'].to_list()[idx+1]))
        pltt['mov_rl'] = [0]+diff 
        #pltt['mov_rl'] = pltt['mov_rl'].apply(lambda x: min(x*100, 100_000)) 
        self.FEATURE_DF['GLOSS-SEGMENTS'] = pltt[pltt.columns[-1]].apply(lambda x: 1 if x > pltt[pltt.columns[-1]].describe()[percentage] else 0)
        
        
        self.FEATURE_DF['GLOSS-SEGMENTS'] = self.FEATURE_DF['GLOSS-SEGMENTS'].rolling(roll).mean().replace(np.nan, 0).apply(lambda x: '0' if x >= threshold else '1')