import os
import sys
import cv2
import asyncio
import streamlit as st
import mediapipe as mp
import hydralit_components as hc
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
from lib.phonology.handshape import *
from lib.phonology.temporal import *
from lib.phonology.classifier import *
from lib.phonology.elan import ELANWriter
from components.navbar import navbar
from components.search import search
from components.extraction import extract
from PIL import Image


#make it look nice from the start
st.set_page_config(layout='wide', initial_sidebar_state='collapsed')

if 'log' not in st.session_state:
    st.session_state.log = None


# specify the primary menu definition
menu_data = [
    {'id': 'feature-search', 'icon':"fa fa-search",'label':"Search"},
    {'id': 'feature-extraction','icon': "far fa-copy", 'label':"Feature Extraction"},
    {'icon': "fas fa-tachometer-alt", 'label':"Dashboard",'ttip':"I'm the Dashboard tooltip!"}, #can add a tooltip message
]

over_theme = {'txc_inactive': '#FFFFFF'}
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    hide_streamlit_markers=True, #will show the st hamburger as well as the navbar now!
    sticky_nav=True, #at the top or not
    sticky_mode='pinned', #jumpy or not-jumpy, but sticky or pinned
)

if menu_id == 'feature-extraction':
    extract()

if menu_id == 'feature-search': 
    search(menu_id)               