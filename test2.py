import glob
from tqdm import tqdm

import numpy as np
from numpy import dot
from numpy.linalg import norm
import os

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

l = glob.glob("precedent/np2/*.npy")
precedent_name_list = [path.split("/")[-1].split('_')[0] for path in l]
# precedent_name_list = ["64443", "67384", "67672"]
# print(len(l))
# print(l[0])
# import sys
# sys.exit()




data_dict = {}



name_list = []
for idx, name1 in tqdm(enumerate(precedent_name_list)):
    name_list.append(name1)
    data_dict[name1] = []

    sub_file_list1 = glob.glob(os.path.join("precedent", "np2", f"{name1}*.npy"))
    vec1_list = np.array([np.load(path) for path in sub_file_list1])
    # print(name1)
    # print(vec1_list.shape)
    # with open(f"precedent/judgment/{name1}.txt", "r") as f:
    #     for i in f.read().strip().split('\n'):
    #         print(i)
    #         print('-----')

    for i in range(idx+1):
        data_dict[name1].append(0)
    
    for name2 in tqdm(precedent_name_list[idx+1:], leave=False):
        sub_file_list2 = glob.glob(os.path.join("precedent", "np2", f"{name2}*.npy"))
        vec2_list = np.array([np.load(path) for path in sub_file_list2])
        # print(name2)
        # print(vec2_list.shape)
        # with open(f"precedent/judgment/{name2}.txt", "r") as f:
        #     for i in f.read().strip().split('\n'):
        #         print(i)
        #         print('-----')

        cs_list = []
        for v1 in vec1_list:
            for v2 in vec2_list:
                cs = cos_sim(v1, v2)
                cs_list.append(cs)
        # print(f"max score: {np.max(cs_list)}")
        data_dict[name1].append(np.max(cs_list))
    #     break
    # break
import pandas as pd
df = pd.DataFrame(data_dict)
df.to_csv("matrix.csv")
print(df.shape)
print("done!")

# p_list = df.columns.to_list()

# id1 = 99
# id2 = np.argmax(df.iloc[2])
# print(p_list[id1])
# print(p_list[id2])

# with open(f"precedent/judgment/{p_list[id1]}.txt", "r") as f:
#     print(f.read())
# print('----')
# with open(f"precedent/judgment/{p_list[id2]}.txt", "r") as f:
#     print(f.read())