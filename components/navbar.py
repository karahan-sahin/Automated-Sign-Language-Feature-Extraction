
import os
import cv2
import streamlit as st

def navbar():
    
    PARAMS = {}
    
    videos = ['Camera'] + ['data/samples/' +
                    file_name for file_name in os.listdir('data/samples')]

    option = st.sidebar.selectbox('Select vids', videos)
    
    if option != 'Camera': PARAMS['file_name'] = option.split('/')[-1].split('.')[0]

    camera = cv2.VideoCapture(0 if option == 'Camera' else option)
    PARAMS['camera'] = camera
    PARAMS['option'] = option
    
    display = st.sidebar.selectbox(
        'Select metric', ('feature', 'boundary', 'classification'), index=2)

    PARAMS['display'] = display

    if display == 'feature':
        selected = st.sidebar.multiselect(
            'Select metric', ('LOCATION', 'ORIENTATION', 'FINGER_SELECTION', 'Path', 'Spacial', 'Temporal', 'Setting'), default=['LOCATION', 'ORIENTATION', 'FINGER_SELECTION', 'Path', 'Spacial', 'Temporal', 'Setting'])

        PARAMS['selected'] = selected
        
        
    if display == 'boundary':

        selected = []
        window_size = st.sidebar.number_input(
            'Select MovAvg window', value=5)
        view_window = st.sidebar.number_input(
            'Select Motion Track window', value=100)

        PARAMS['window_size'] = window_size
        PARAMS['view_window'] = view_window

        hand = st.sidebar.selectbox(
            'Select hand for movement tracking', ('BOTH', 'LEFT', 'RIGHT'))
        
        PARAMS['selected'] = selected

        if not hand == 'BOTH':
            metrics = st.sidebar.multiselect(label='Select Metrics',
                                            options=['Center_Diff', 'Center_MovAVG_x', 'Center_MovAVG_y', 'Center_MovAVG_z',
                                                    'INDEX_FINGER', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY', 'SWITCH', 'LSE_x', 'LSE_y'],
                                            default=['Center_Diff', 'Center_MovAVG_x', 'Center_MovAVG_y', 'Center_MovAVG_z', 'INDEX_FINGER', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY', 'SWITCH', 'LSE_x', 'LSE_y'])

            PARAMS['metrics'] = metrics

        else:
            metrics = st.sidebar.multiselect(label='Select Metrics',
                                            options=['Center_MovAVG_Left', 'Center_MovAVG_Right', 'Aperture_MovAVG_Left',
                                                    'Aperture_MovAVG_Right', 'SWITCH_Left', 'SWITCH_Right'],
                                            default=['Center_MovAVG_Left', 'Center_MovAVG_Right', 'Aperture_MovAVG_Left', 'Aperture_MovAVG_Right', 'SWITCH_Left', 'SWITCH_Right'])

            PARAMS['metrics'] = metrics

    elif display == 'classification':

        selected = st.sidebar.multiselect(
            'Select metric', ('LOCATION', 'ORIENTATION', 'FINGER_SELECTION', 'Path', 'Spacial', 'Temporal', 'Setting'), default=['LOCATION', 'ORIENTATION', 'FINGER_SELECTION', 'Path', 'Spacial', 'Temporal', 'Setting'])

        
        PARAMS['selected'] = selected
        PARAMS['topK'] = st.sidebar.number_input(
            'Select Top Predictions', value=5)
        
        PARAMS['model_name'] = st.sidebar.selectbox(
                'Select Model', ('baseline', 'LSTM Auto-encoder', 'Transformer Auto-encoder'))
        PARAMS['timesteps'] = st.sidebar.slider( 'Select a range of timesteps', 1, 60, (1, 60))


    run = st.sidebar.button('Run', key=1)
    PARAMS['run'] = run

    return PARAMS