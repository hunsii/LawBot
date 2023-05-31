"""
from sentence_transformers import SentenceTransformer
sentences = ["안녕하세요?", "한국어 문장 임베딩을 위한 버트 모델입니다."]

model = SentenceTransformer('jhgan/ko-sbert-multitask')
embeddings = model.encode(sentences)
print(embeddings.shape)
"""

from transformers import AutoTokenizer, AutoModel
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

import glob
import os
import random
def get_sentences(n = 2):
    precedent_n_list = glob.glob("판례/판시사항/*.txt")
    precedent_n_list = [os.path.basename(path) for path in precedent_n_list]
    print(f"판례 수: {len(precedent_n_list)}")

    sentence_list = []
    for i in range(n):
        rnd_idx = random.randrange(len(precedent_n_list))
        rnd_precedent_n = precedent_n_list[rnd_idx]
        with open(f"판례/판례내용/{rnd_precedent_n}", "r") as f:
            data = f.read()

        sentence_list.append(data.replace("\n", " "))
        print(data)
    return sentence_list

# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']
sentences = get_sentences()

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('jhgan/ko-sbert-multitask')
model = AutoModel.from_pretrained('jhgan/ko-sbert-multitask')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, mean pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

print("Sentence embeddings:")
print(sentence_embeddings.shape)