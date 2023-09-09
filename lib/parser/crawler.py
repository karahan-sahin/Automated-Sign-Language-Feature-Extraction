import os
import re
import math
import requests
from urllib.request import urlopen
from tqdm import tqdm
from lxml import html
import os
from tqdm import tqdm


def groupById(vids: list):
    GROUPS = {}
    curr_id = ''
    for vid in vids:
        _,_,_id,_,fname = vid.split('/')
        if _id != curr_id:
            GROUPS[_id] = [vid]
            curr_id = _id
        else: GROUPS[curr_id].append(vid)
    return GROUPS

CHARS = ['A', 'B', 'C', 'Ç', 'D', 'E', 'F', 'G', 'H', 'I', 'İ', 'J', 'K', 'L', 'M', 'N', 'O', 'Ö', 'P', 'R', 'S', 'Ş', 'T', 'U', 'Ü', 'V', 'W', 'Y', 'Z']
ROW_XPATH = 
for char in tqdm(CHARS):
    
    print(f'At {char=}')
    
    page = str(1)
    BASE_URL = 'https://tidsozluk.aile.gov.tr'
    SEARCH_URL = f'https://tidsozluk.aile.gov.tr/tr/Alfabetik/Arama/{char}?p={page}'
    
    HTML = html.fromstring(requests.get(SEARCH_URL).content)
    PAGES = math.ceil(int(HTML.xpath('//div[contains(@id, "rezults_summ")]//b/text()')[0]) / 10)
    
    for page in range(1, PAGES+1):    
        
        print(f'At {page=}')
        
        if page > 1:
            HTML = html.fromstring(requests.get(f'https://tidsozluk.aile.gov.tr/tr/Alfabetik/Arama/{char}?p={page}').content)
    
        LEXICON = [res.xpath('//h3/text()') for res in HTML.xpath(ROW_XPATH)][0]
        
        VIDEOS = [ re.sub('0\.1','0.5', vid) for vid in [res.xpath('//source/@src') for res in HTML.xpath(ROW_XPATH)][0]]
    
        for name, vids in tqdm(list(zip(LEXICON,list(groupById(VIDEOS).values())))):
            
            for idx, vid in enumerate(vids):
                
                f = urlopen(BASE_URL+vid)
                with open(f'../data/corpus/{name.upper()}_{idx}.mp4', 'wb') as code:
                    code.write(f.read())