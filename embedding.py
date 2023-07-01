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
os.makedirs("precedent/np/", exist_ok=True)

l = glob.glob("precedent/judgment/*.txt")
n_null_count = 0
precedent_embedding_dict = {}
for path in tqdm(l):
    name = path.split("/")[-1].split('.')[0]
    with open(path, 'r') as f:
        data = f.read().strip()
    sentences = []
    for line in data.splitlines():
        if not line.startswith(" "):
            sentences.append([])
            sentences[-1].append(line)
    sentences = [sentence[0] for sentence in sentences]

    if len(sentences) == 0:
        n_null_count += 1
        continue
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, mean pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # print("Sentence embeddings:")
    # print(sentence_embeddings.shape)
    # print(sentence_embeddings)

    # print(sentence_embeddings.squeeze().numpy().shape)
    # for idx, sentence_embedding in enumerate(sentence_embeddings):
    sentence_numpy = sentence_embeddings.numpy()
    np.save(os.path.join("precedent", "np", f"{name}.npy"), sentence_numpy)
    precedent_embedding_dict[name] = sentence_numpy
    # break

import pickle
with open("precedent_embedding_dict.pickle","wb") as f:
    pickle.dump(precedent_embedding_dict, f) # 위에서 생성한 리스트를 list.pickle로 저장   
print(n_null_count) # 13543
print("done!")