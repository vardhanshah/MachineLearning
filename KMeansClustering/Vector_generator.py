import random
import numpy as np
def generate(filename,vec_len,num_vecs):
    with open(filename,'w') as f:
        print(vec_len,num_vecs)
        for i in range(num_vecs):
            vec=[]
            for j in range(vec_len):
                vec.append(np.random.randint(ll,ul))
            for k in vec:
                f.write(str(k) + ' ')
            f.write('\n')
ll = 1
ul = 1000
filename1 = "vectors"
filename2 = "candid_vectors"
vec_len = 2
num_vecs = 1000
num_candid_vecs = 3000

generate('vectors',vec_len,num_vecs)
generate('candid_vectors',vec_len,num_candid_vecs)
