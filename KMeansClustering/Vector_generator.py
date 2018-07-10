import random
import numpy as np
def generate(filename,vec_len,num_vecs):
    with open(filename,'w') as f:
        print(vec_len,num_vecs)
        for i in range(num_vecs):
            vec=[]
            for j in range(vec_len):
                vec.append(np.random.randint(ll,ul))
            print(len(vec),vec_len)
            for k in vec:
                f.write(str(k) + ' ')
            f.write('\n')
lower_limit = 1
upper_limit = 1000
filename = "vectors"
vec_len = 2
num_vecs = 500
num_candid_vecs = 1000

generate('vectors',vec_len,num_vecs)
generate('candid_vector',vec_len,num_candid_vecs)
