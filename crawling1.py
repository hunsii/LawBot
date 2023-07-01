import pandas as pd
import xml.etree.ElementTree as ET
from urllib.request import urlopen
from tqdm import trange
import time


id = "YOUR-ID-HERE"

url = f"https://www.law.go.kr/DRF/lawSearch.do?OC={id}&target=prec&type=XML&page=0"
# response = urlopen(url).read()
# xtree = ET.fromstring(response)
# totalCnt = int(xtree.find('totalCnt').text)
totalCnt = 85610


columns = [
    '판례일련번호', 
    '사건명', 
    '사건번호', 
    '선고일자', 
    '법원명', 
    '법원종류코드', 
    '사건종류명', 
    '사건종류코드', 
    '판결유형', 
    '선고', 
    '판례상세링크', 
]

import os
start_idx = 0
if os.path.exists('temp.txt'):
    with open('temp.txt', 'r') as f:
        start_idx = int(f.read())
        print(f'start idx: {start_idx}')

df_list = []
for page_idx in trange(start_idx, int(totalCnt/20)):
    url = f"https://www.law.go.kr/DRF/lawSearch.do?OC={id}&target=prec&type=XML&page={page_idx}"
    response = urlopen(url).read()
    xtree = ET.fromstring(response)
    
    for idx, node in enumerate(xtree[5:]): # 앞부분 생략
        data_dict = {}
        for c in columns:
            data_dict[c] = node.find(c).text
        df_list.append(data_dict)

    with open('temp.txt', 'w') as f:
        f.write(str(page_idx))
    
    new_df = pd.DataFrame(df_list, columns=columns)
    new_df.to_csv('data.csv', index=False)
    
print('done!')