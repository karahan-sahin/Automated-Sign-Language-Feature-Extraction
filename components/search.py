import streamlit as st
from components.navbar import navbar
import pandas as pd
import numpy as np
from scipy.spatial import distance

def search(page_id):
    
    print(page_id)
    
    PARAMS = navbar(page_id)
    
    print(PARAMS)

    video_file = open('data/corpus/'+PARAMS['file_name']+'.mp4', 'rb')
    video_bytes = video_file.read()
    
    st.sidebar.video(video_bytes)

    width = st.sidebar.slider(
        label="Width", min_value=0, max_value=100, value=80, format="%d%%"
    )

    width = max(width, 0.01)
    side = max((100 - width) / 2, 0.01)

    if PARAMS['run']:
        
        X = np.load(open('models/feature-corpus.npy', 'rb'))
        LEXICON = open('models/lexicon.txt', 'r', encoding='utf-8').read().split('\n')
        idx2lex = dict(enumerate(LEXICON))
        lex2idx = {j:i for i,j in idx2lex.items()}

        

        dist = {}
        for ix, lex in enumerate(LEXICON):
            dist[lex] = distance.euclidean(X[lex2idx[PARAMS['file_name']], :], X[ix, :])
            
        for lex, score in pd.Series(dist).sort_values().head(10).to_dict().items():

            st.write('Lexicon:', lex)
            st.write('Score:', score)
            
            _, container, _ = st.columns([side, width, side])
            video_file = open(f'data/corpus/{lex}.mp4', 'rb')
            video_bytes = video_file.read()

            container.video(video_bytes)
