"""
특정 상황을 입력하면, 
판례 데이터 중에서 그 상황과 가장 유사한 판례를 찾아주는 코드입니다.
"""

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import os

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('jhgan/ko-sroberta-multitask')
model = AutoModel.from_pretrained('jhgan/ko-sroberta-multitask')

import glob
from tqdm import tqdm

sentence = "약혼이 해제되어 파혼될 경우에 약혼예물로 주고받은 물건에 대해 따로 합의한 바가 없다면, 어떻게 처리해야 하나요?"
sentences = sentence.split("\n")

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, mean pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
vecs1 = sentence_embeddings.numpy()

import pickle
import glob
import numpy as np
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm

MAX_RESULT = 3

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

with open("precedent_embedding_dict.pickle","rb") as f:
    precedent_dict = pickle.load(f)

precedent_name_list = list(precedent_dict.keys())

print(f"입력한 사례\n{sentence}")
print('-'*80)

cs_list = []

for key2, vecs2 in tqdm(precedent_dict.items(), leave=False):
    sub_list = [-1]
    for v1 in vecs1:
        for v2 in vecs2:
            cs = cos_sim(v1, v2)
            sub_list.append(cs)
    cs_list.append(np.max(sub_list))

cs_list = np.array(cs_list)

# 가장 큰 값과 그 값의 인덱스 뽑아내기
sorted_indices = np.argsort(cs_list)
largest_values = cs_list[sorted_indices[-10:]][::-1]
largest_indices = sorted_indices[-10:][::-1]

for i in range(10):
    # print(i)
    similar_precedent_score = largest_values[i]#np.max(cs_list)
    similar_precedent_index = largest_indices[i]#np.argmax(cs_list)
    similar_precedent_number = precedent_name_list[similar_precedent_index]
    
    with open(f"precedent/judgment/{list(precedent_name_list)[similar_precedent_index]}.txt", "r") as f:
        similar_precedent = f.read()
    
    if similar_precedent == '':
        continue
    print(f"유사한 판례: {similar_precedent_number}, {similar_precedent_score:.2f}\n{similar_precedent}")
    print("- - - "*10)
    if i >= MAX_RESULT:
        break
    # print('-'*80)