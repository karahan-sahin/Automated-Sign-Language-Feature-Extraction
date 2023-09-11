import numpy as np
from ..utils._utils import toArray
from ..phonology.sign_utils import COLS, MODEL_DIR
from ..utils._math import cosine_similarity_numba

from typing import List
class LexicalClassification():
    
    def __init__(self, params) -> None:        
        self.X = None
        self.IDX2LABEL = None
        self.LABEL2IDX = None
        
        self.PREFIX = MODEL_DIR[params['model_name']]
        
        if params['model_name'] == 'baseline':
            self.start_idx, self.end_idx = params['timesteps'][0], params['timesteps'][1]
            self.loadLabels()
            self.load3DTrainset(self.start_idx, self.end_idx, True)
        
        self.LEXICON_MEMORY = []
        
    def load3DTrainset(self, 
                       start_idx: int,
                       end_idx: int, 
                       flatten: bool) -> None: 
        
        self.X = np.load(f'{self.PREFIX}/vectors.npy')
        # Add for max_seq_length for 2D
        self.X = self.X[:,start_idx-1:end_idx,:]
        # If you 2D dataset
        self.X = self.X.reshape(self.X.shape[0], self.X.shape[1]*len(COLS))
        # Change type for numba cosine implementation
        self.X = self.X.astype('float32')
    
    def loadLabels(self) -> None:
        LABELS = open(f'{self.PREFIX}/labels.txt').read().split('\n')[:-1]
        self.IDX2LABEL = dict(enumerate(LABELS))
        self.LABEL2IDX = { label: idx for idx, label in self.IDX2LABEL.items() }
        
    def transform(self, HIST):
        if len(HIST[-self.end_idx-1 :-self.start_idx]) > (self.end_idx - self.start_idx):
            return toArray(HIST[-self.end_idx-1 :-self.start_idx]).reshape(-1).astype('float32')
        return

    async def predict(self, 
                      A: np.ndarray, 
                      topK: int) -> List[ tuple[str, int] ]:
        
        
        if A is None: return
        
        scores = {}
        for idx, lex in self.IDX2LABEL.items():
            B = self.X[idx]
            scores[lex] = cosine_similarity_numba(A, B)

        topK_results = sorted(scores.items(),key=lambda x: x[1], reverse=True)[:topK]
        
        # Add results to memory
        results = {}
        results.update({f'LABEL_{rank}': lex for rank, (lex, score) in enumerate(topK_results)})
        results.update({f'SCORE_{rank}': score for rank, (lex, score) in enumerate(topK_results)})
        self.LEXICON_MEMORY.append(results)
        
        return topK_results