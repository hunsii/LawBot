import pickle
import glob
import numpy as np
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

l = glob.glob("precedent/judgment/*.txt")
precedent_name_list = [path.split("/")[-1].split('.')[0] for path in l]
with open("precedent_embedding_dict.pickle","rb") as f:
    precedent_dict = pickle.load(f)
target_precedent = ""
while target_precedent == "":
    random_number = np.random.randint(1, len(precedent_name_list))
    target_precedent_number = precedent_name_list[random_number]
    with open(f"precedent/judgment/{target_precedent_number}.txt", "r") as f:
        target_precedent = f.read()
print(f"선택한 판례: {target_precedent_number}\n{target_precedent}")
print('-'*80)

vecs1 = precedent_dict[target_precedent_number]
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

# print("가장 큰 값 배열:", largest_values)
# print("인덱스 배열:", largest_indices)
# 배열에서 가장 큰 값 3개를 뽑아내기
# largest_values = np.partition(cs_list, -10)[-10:]

# 배열에서 가장 큰 값 10개의 인덱스를 알아내기
# largest_indices = np.argpartition(cs_list, -10)[-10:]


# print(largest_values)
# print(largest_indices)


# cs_list.pop(random_number)
for i in range(10):
    # print(i)
    similar_precedent_score = largest_values[i]#np.max(cs_list)
    similar_precedent_index = largest_indices[i]#np.argmax(cs_list)
    # print(similar_precedent_index)
    # print(np.max(cs_list))
    # print(np.argmax(cs_list))
    similar_precedent_number = precedent_name_list[similar_precedent_index]
    # print(target_precedent_number)
    # print(similar_precedent_number)
    if target_precedent_number == similar_precedent_number:
        continue
    # print(len*l)
    # break
    with open(f"{l[int(similar_precedent_index)]}", "r") as f:
        similar_precedent = f.read()
    
    if similar_precedent == '':
        continue
    # print(similar_precedent_number)
    # print(similar_precedent_score)
    # print(similar_precedent)
    print(f"유사한 판례: {similar_precedent_number}, {similar_precedent_score:.2f2}\n{similar_precedent}")
    break
    # print('-'*80)