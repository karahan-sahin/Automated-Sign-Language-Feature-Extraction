

class SignBoundarySegmentation():
    
    def __init__(self, ) -> None: pass
    
    
    def parseSegments(self, 
                      percentage='75%', 
                      roll=10):

    pltt = self.FEATURE_DF[NUMERICAL] * 1000

    pltt = pltt.replace(np.nan, 0)

    mov = pd.Series(np.zeros(pltt.shape[0]))

    for col in NUMERICAL:
        mov+=(pltt[col].rolling(roll).mean()) / 10
        
    pltt['mov'] = mov 
    diff = []
    for idx, dd in enumerate(pltt['mov'].to_list()[:-1]):
        diff.append(abs(dd - pltt['mov'].to_list()[idx+1]))
    pltt['mov_rl'] = [0]+diff 
    #pltt['mov_rl'] = pltt['mov_rl'].apply(lambda x: min(x*100, 100_000)) 
    self.FEATURE_DF['GLOSS-SEGMENTS'] = pltt[pltt.columns[-1]].apply(lambda x: 1 if x > pltt[pltt.columns[-1]].describe()[percentage] else 0)
    
    
    self.FEATURE_DF['GLOSS-SEGMENTS'] = self.FEATURE_DF['GLOSS-SEGMENTS'].rolling(2).mean().replace(np.nan, 0).apply(lambda x: '0' if x > 0.5 else '1')

    
    
    