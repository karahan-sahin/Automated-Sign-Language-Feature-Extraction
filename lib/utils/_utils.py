from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import numpy as np
import pandas as pd
from ..phonology.sign_utils import COLS

def toArray(INFO):
    
    data_p = pd.DataFrame.from_records(INFO)
    data_p.replace(True, 1, inplace=True)
    data_p.replace(False, 0, inplace=True)

    categorical_cols = [i for i in data_p.columns if i.endswith('LABEL')]
    data_p = data_p.fillna(data_p.mean(numeric_only=True).round(1))
    for col in categorical_cols:
        le = LabelEncoder()
        data_p[col] = le.fit_transform(data_p[col])
    
    data_p = data_p.fillna(data_p.mode(numeric_only=True).round(1))
    data_p.fillna(0, inplace=True)

    for col in COLS:
        if col in data_p.columns: 
            data_p[col] = data_p[col].astype('float128')
        else:
            data_p[col] = 0
            data_p[col] = data_p[col].astype('float128')
                
    return data_p[COLS].to_numpy()
    
    
    
    