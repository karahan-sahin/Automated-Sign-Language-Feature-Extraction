
import os
import cv2
import streamlit as st


def navbar(page):

    PARAMS = {}

    if page == 'feature-extraction':

        source_exp = st.sidebar.expander("File Source", expanded=True)

        def fetchFile(fname, idx):

            idx += 1
            folders = [a for a in os.listdir(fname) if a != '.DS_Store']
            SUB_FOLDER = source_exp.selectbox(
                f'Select data source - Subfolder {idx}', folders)
            if SUB_FOLDER.split('.')[-1] != 'mp4':
                return fetchFile(fname+SUB_FOLDER+'/', idx)
            else:
                return fname+SUB_FOLDER

        option = fetchFile('data/', 0)

        if option != 'Camera':
            PARAMS['file_name'] = option.split('/')[-1].split('.')[0]

        camera = cv2.VideoCapture(0 if option == 'Camera' else option)
        PARAMS['camera'] = camera
        PARAMS['option'] = option

        model_exp = st.sidebar.expander("Model Parameters", expanded=True)

        display = model_exp.selectbox(
            'Select metric', ('feature', 'classification'), index=0)

        PARAMS['display'] = display

        PARAMS['percentage'] = model_exp.selectbox(
            f'Select Segment Percentage', ['50%', '75%', 'mean', 'std'])
        PARAMS['roll'] = model_exp.selectbox(
            f'Select Roll window', [1, 2, 5, 7, 10])

        threshold = model_exp.number_input(
            'Give Segment Threshold', min_value=0.1, max_value=0.9)
        PARAMS['threshold'] = threshold

        selected = model_exp.multiselect(
            'Select metric', ('LOCATION', 'ORIENTATION', 'FINGER_SELECTION', 'Path', 'Spacial', 'Temporal', 'Setting'), default=['LOCATION', 'ORIENTATION', 'FINGER_SELECTION', 'Path', 'Spacial', 'Temporal', 'Setting'])

        PARAMS['selected'] = selected

        if display == 'boundary':

            selected = []
            window_size = model_exp.number_input(
                'Select MovAvg window', value=5)
            view_window = model_exp.number_input(
                'Select Motion Track window', value=100)

            PARAMS['window_size'] = window_size
            PARAMS['view_window'] = view_window

            hand = model_exp.selectbox(
                'Select hand for movement tracking', ('BOTH', 'LEFT', 'RIGHT'))

            PARAMS['selected'] = selected

            if not hand == 'BOTH':
                metrics = model_exp.multiselect(label='Select Metrics',
                                                options=['Center_Diff', 'Center_MovAVG_x', 'Center_MovAVG_y', 'Center_MovAVG_z',
                                                         'INDEX_FINGER', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY', 'SWITCH', 'LSE_x', 'LSE_y'],
                                                default=['Center_Diff', 'Center_MovAVG_x', 'Center_MovAVG_y', 'Center_MovAVG_z', 'INDEX_FINGER', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY', 'SWITCH', 'LSE_x', 'LSE_y'])

                PARAMS['metrics'] = metrics

            else:
                metrics = model_exp.multiselect(label='Select Metrics',
                                                options=['Center_MovAVG_Left', 'Center_MovAVG_Right', 'Aperture_MovAVG_Left',
                                                         'Aperture_MovAVG_Right', 'SWITCH_Left', 'SWITCH_Right'],
                                                default=['Center_MovAVG_Left', 'Center_MovAVG_Right', 'Aperture_MovAVG_Left', 'Aperture_MovAVG_Right', 'SWITCH_Left', 'SWITCH_Right'])

                PARAMS['metrics'] = metrics

        elif display == 'classification':

            selected = model_exp.multiselect(
                'Select metric', ('LOCATION', 'ORIENTATION', 'FINGER_SELECTION', 'Path', 'Spacial', 'Temporal', 'Setting'), default=['LOCATION', 'ORIENTATION', 'FINGER_SELECTION', 'Path', 'Spacial', 'Temporal', 'Setting'])

            PARAMS['selected'] = selected
            PARAMS['topK'] = model_exp.number_input(
                'Select Top Predictions', value=5)

            PARAMS['model_name'] = model_exp.selectbox(
                'Select Model', ('baseline', 'LSTM Auto-encoder', 'Transformer Auto-encoder'))
            PARAMS['timesteps'] = model_exp.slider(
                'Select a range of timesteps', 1, 60, (1, 60))

        run = model_exp.button('Run', key=1)
        PARAMS['run'] = run

        PARAMS['export_type'] = st.sidebar.selectbox(
            'Select export option', ('Eaf File', 'Excel File'))
        PARAMS['stop'] = st.sidebar.button('Save Live Stream', key=2)

    if page == 'feature-search':

        feats = []
        
        search_type = st.sidebar.selectbox('Select Search by:', [
                                'feature', 'video'])
        
        if search_type == 'video':
            
            source_exp = st.sidebar.expander("File Source", expanded=True)
            idx = 0
            def fetchFile(fname, idx):

                idx += 1
                folders = [a for a in os.listdir(fname) if a != '.DS_Store']
                SUB_FOLDER = source_exp.selectbox(
                    f'Select data source - Subfolder {idx}', folders)
                if SUB_FOLDER.split('.')[-1] != 'mp4':
                    return fetchFile(fname+SUB_FOLDER+'/', idx)
                else:
                    return fname+SUB_FOLDER

            option = fetchFile('data/', 0)

            PARAMS['file_name'] = option.split('/')[-1].split('.')[0]
                
            PARAMS['run'] = st.sidebar.button('run')
        
        elif search_type == 'feature':

            ncol = st.sidebar.number_input("Number of dynamic columns", 0, 20, 1)

            for col in range(ncol):

                exp = st.sidebar.expander(f'Feature {col}')

                feat = exp.selectbox('Select Feature:', [
                                    'LOCATION', 'ORIENTATION', 'FINGER_SELECTION', 'MANNER'], key=col)

                if feat == 'LOCATION':
                    hand = exp.multiselect('Select Hand:', ('LEFT', 'RIGHT'), default=[
                                        'LEFT', 'RIGHT'], key=f'{col}_hand')
                    
                    loc = exp.selectbox('Select Major Location', ( 'HIP','CHEST','ARM','SHOULDER','EAR','MOUTH','EYE','NOSE'), key=f'{col}_loc' )


                if feat == 'FINGER_SELECTION':
                    hand = exp.multiselect('Select Hand:', ('LEFT', 'RIGHT'), default=[
                                        'LEFT', 'RIGHT'], key=f'{col}_hand')
                    select = exp.multiselect('Select Selection Type:', ('SELECT', 'CURVED'), default=[
                                            'SELECT', 'CURVED'], key=f'{col}_select')
                    finger = exp.multiselect('Select Fingers:', ('THUMB', 'INDEX_FINGER', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY'), default=[
                                            'THUMB', 'INDEX_FINGER', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY'], key=f'{col}_finger')
                    angle = exp.multiselect('Select Angle Type:', ('PIP', 'DIP', 'WRST_PIP', 'WRST_DIP'), default=[
                                            'PIP', 'DIP', 'WRST_PIP', 'WRST_DIP'], key=f'{col}_angle')

                if feat == 'ORIENTATION':
                    hand = exp.multiselect('Select Hand:', ('LEFT', 'RIGHT'), default=[
                                        'LEFT', 'RIGHT'], key=f'{col}_hand')
                    
                    finger = exp.multiselect('Select Type:', ('PALM/BACK', 'TIPS/WRIST', 'RADIAL/URNAL'), default=[
                                            'PALM/BACK', 'TIPS/WRIST', 'RADIAL/URNAL'], key=f'{col}_finger')
                    
                    for feat in finger:
                        feats = feat.lower().split('/')
                        angle = exp.selectbox('Select Angle Type:', feats, key=f'{col}_{feat}')
                        
                if feat == 'MANNER':
                    feats = exp.multiselect('Select Type:', ('Path', 'Spacial', 'Temporal', 'Setting'), default=[
                                            'Path', 'Spacial', 'Temporal', 'Setting'], key=f'{col}_manner')
                    
                    feat_dict = {
                        'Path': ['straight', 'curved'],
                        'Setting': ['High', 'Low'],
                        'Spacial': {
                            'IN_FRONT': ['left', 'right'],
                            'ABOVE/BELOW': ['left', 'right'],
                            'INTERWOVEN': ['TRUE', 'FALSE'],
                        },
                        'Temporal': {
                            'REPETITION': ['TRUE', 'FALSE'],
                            'ALTERNATION': ['TRUE', 'FALSE'],
                            'CONTACT': ['TRUE', 'FALSE'],
                            'SYNC/ASYNC': ['TRUE', 'FALSE'],
                        }
                    }
                    
                    for feat in feats:
                        if isinstance(feat_dict[feat],list):
                            feats = exp.selectbox(f'Select {feat} Label:', feat_dict[feat], key=f'{col}_{feat}')

                        if isinstance(feat_dict[feat],dict):
                            sub_feats = exp.multiselect(f'Select {feat} Type:', list(feat_dict[feat].keys()), default= list(feat_dict[feat].keys()), key=f'{col}_{feat}')                    
                            for sub_type in sub_feats:
                                sub_type_select = exp.selectbox(f'Select {feat} Label:', feat_dict[feat][sub_type], key=f'{col}_{feat}_{sub_type}')


    return PARAMS
