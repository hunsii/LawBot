import pandas as pd
from tqdm.auto import tqdm
from urllib.request import urlopen
import xml.etree.ElementTree as ET
import os
df = pd.read_csv("data.csv")
print(df.columns)
print('-'*80)
# print(df.head())
from bs4 import BeautifulSoup
import requests

# 웹 User-Agent
web_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"

# 웹 페이지 요청 - 모바일 버전으로 시도
# response = requests.get(url, headers={"User-Agent": mobile_user_agent})

columns = [
    '판례정보일련번호',   # 중복
    '사건명',          # 중복
    '사건번호',         # 중복
    '선고일자',         # 중복, 형식 다름
    '선고',            # 중복
    '법원명',          # 중복
    '법원종류코드',      # 중복, 하나는 정보 없음
    '사건종류명',       # 중복
    '사건종류코드',     # 중복
    '판결유형',       # 중복
    '판시사항', 
    '판결요지', 
    '참조조문', 
    '참조판례', 
    '판례내용', 
]
os.makedirs("판례/판시사항", exist_ok=True)
os.makedirs("판례/판결요지", exist_ok=True)
os.makedirs("판례/참조조문", exist_ok=True)
os.makedirs("판례/참조판례", exist_ok=True)
os.makedirs("판례/판례내용", exist_ok=True)
import re
def remove_tag(content):
    cleaned_text = re.sub('<.*?>', '', content)
    return cleaned_text

data_list = []
url_default = "https://www.law.go.kr"

for idx in tqdm(df.index):
    if idx < 78840:
        continue
    url_detail = df.loc[idx, "판례상세링크"].replace('HTML', 'XML')
    url = f"{url_default}{url_detail}"

    response = urlopen(url).read()
    xtree = ET.fromstring(response)

    # for idx, node in enumerate(xtree):
    data_dict = {}
    for c in columns:
        text = xtree.find(c).text
        if text is not None:
            data_dict[c] = remove_tag(text.replace("<br/>", "\n"))
        else:
            data_dict[c] = ""
        # except:
        #     data_dict[c] = ""
        #     # print(f"{data_dict['판례정보일련번호']} : {c}")
        #     print(xtree.find(c).text)
        #     raise("eee")
    # data_list.append(data_dict)
    for c in [    '판시사항', 
    '판결요지', 
    '참조조문', 
    '참조판례', 
    '판례내용', ]:
        with open(f"판례/{c}/{data_dict['판례정보일련번호']}.txt", "w") as f:
            if data_dict[c] != "":
                f.write(data_dict[c])
        
    # break
# new_df = pd.DataFrame(data_list)
# new_df.to_csv("precedent.csv", index=False)
# print("done!")