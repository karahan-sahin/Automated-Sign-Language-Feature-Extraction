import cv2
import pandas as pd
from pympi.Elan import Eaf
from .sign_utils import FEAT_TIER_MAP


class ELANWriter():

    def __init__(self, feature_df, video_path) -> None:

        self.FEATURE_DF = feature_df
        self.video_file_path = video_path

        self.ELAN = Eaf(author='pympi')
        self.createFile(['FINGER_COORDINATION', 'MANNER', 'ORIENTATION', 'LOCATION'])

        self.VIDEO_LENGTH = None
        self.VIDEO_FPS = None
        
        self.getVideoProperties()

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
                        if seg_label != None:
                            try:
                                self.ELAN.add_annotation(
                                    tier_name,
                                    seg_start,
                                    idx-1,
                                    value=seg_label,
                                    svg_ref=None)
                            except ValueError:
                                print('Cannot write annotation:',
                                    seg_label, seg_start, idx-1)

                        seg_start = idx
                        seg_label = label

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
                    f'{HAND}_FINGER_COORDINATION_CURVED', ling='default-lt', parent=f'{HAND}_FINGER_COORDINATION')

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
