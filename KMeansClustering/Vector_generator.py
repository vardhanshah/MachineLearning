import random
import numpy as np
func_rand = np.random.random
ll = 1
ul = 1000
filename = "vectors"
vec_len = 2
num_vecs = 5000

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
generate('vectors',2,1000)
generate('candid_vector',2,1000)
