import random
import time
import sys

no_plot=False

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors
except :
    no_plot=True
    print('No matplotlib is detected, won\'t able to plot')

c= []
bad_colors = ['whitesmoke','floralwhite','lightgoldenrodyellow','aliceblue','plum','w','lightgray','seashell','navajowhite','ivory','lemonchiffon','ghostwhite','mediumturquiose','mediumaquamarine','deeppink','white','dimgray','lightgrey','beige','honeydew','lavender','snow','linen','cornsilk','lightyellow','mintcream','lightcyan']

for name in matplotlib.colors.cnames.keys():
    if name in bad_colors:
        continue
    c.append(name)

#reading config file

filename='config'
config = {}
#params list will contain all the parameters info
params = []
with open(filename,'r') as file:
    for line in file:
        if line=='\n' or line=='' or line[0]=='#' :
            continue
        if not '=' in line:
            print(' \'=\' assignment operator is not found in config file\nplease format the config file properly')
            sys.exit(0)
        k,v = line.split('=')
        rm = ['\n','\t',' ']
        for i in rm:
            k = k.strip(i)
            v = v.strip(i)
        params.append(k)
        config[k] = v
file_vectors = config[params[0]]
file_candid_vectors = config[params[1]]
output_file = config[params[2]]
min_cluster_size = int(config[params[3]])
mx_num_of_sub_clusters = int(config[params[4]])


if len(config)>=7:
    mean_verbose = bool(config[params[5]].title()=='True')
    plot_verbose= bool(config[params[6]].title()=='True')
else:
    mean_verbose=False
    plot_verbose=False
if no_plot or ((mean_verbose==True or plot_verbose==True) and mx_num_of_sub_clusters>len(c)) :
    print('Mean and Cluster plotting cannot be plotted')
    mean_verbose=False
    plot_verbose=False


if len(config)>=8:
    no_of_times = int(config[params[7]])
else:
    no_of_times = 1000
if len(config)>=9:
    init_times = int(config[params[8]])
else:
    init_times = 1
#finds the distance between two vectors

def distance(a,b,sqr_or_not=False):
    dist=0
    for i in range(len(a)):
        dist+=(a[i]-b[i])**2
    if sqr_or_not==True:
        return dist
    return dist**(1/2)


#initializes mean for

def means_init(k,vectors):
    means = [vectors[random.randint(0,len(vectors)-1)]]
    means_len = 1
    while means_len!=k:
        mx_dist_for_mean = 0
        idx=0
        for i in range(len(vectors)):
            mn_dist = distance(vectors[i],means[0],True)
            for j in range(1,means_len):
                dist = distance(vectors[i],means[j],True)
                if dist < mn_dist:
                    mn_dist =dist
            if mn_dist > mx_dist_for_mean:
                mx_dist_for_mean = mn_dist
                idx = i
        means.append(vectors[idx])
        means_len+=1
    return means

#finds the nearest vector for candid_vector, from given vectors

def find_nearest(vectors,candid_vector):
    if len(vectors)==1:
        return vectors
    dist = []
    for i in range(len(vectors)):
        dist.append(distance(vectors[i],candid_vector))
    nearest_vecs = []
    mn = min(dist)
    for i in range(len(vectors)):
        if mn == dist[i]:
            nearest_vecs.append(vectors[i])
    return nearest_vecs

def plot_means(means):
    for i in range(len(means)):
        plt.scatter(means[i][0],means[i][1],c=c[i])



class cluster:

    def __init__(self,vectors,mean,level=0,cluster_no=0):
        self.level = level
        self.cluster_no = cluster_no
        self.vectors = vectors
        self.no_vectors = len(self.vectors)
        self.vec_length = len(self.vectors[0])
        self.mean = mean
        self.no_of_sub_clusters = 0
        self.sub_clusters = []
        self.end = False
        if self.no_vectors < min_cluster_size:
            self.end=True


    def Recursive_search(self,candid_vector,ans):
        # cluster.plot_cluster(self)
        if self.end==True:
            ans.extend(find_nearest(self.vectors,candid_vector))
            return
        dist = []
        for i in range(self.no_of_sub_clusters):
            dist.append(distance(self.sub_clusters[i].mean,candid_vector))
        mn_dist = min(dist)
        for i in range(len(dist)):
            if mn_dist==dist[i]:
                cluster.Recursive_search(self.sub_clusters[i],candid_vector,ans)

    def search(self,candid_vector):
        ans = []
        self.Recursive_search(candid_vector,ans)
        return find_nearest(ans,candid_vector)

    def cost(self,means):
        J=0
        for i in range(self.no_of_sub_clusters):
            for j in range(self.no_vectors):
                J += distance(means[i],vectors[j])
        return J
    def K_Mean(self):
        global min_cluster_size
        global no_of_times
        global init_times
        global mean_verbose
        global plot_verbose


        vec_length = self.vec_length
        no_vectors = self.no_vectors

        k=mx_num_of_sub_clusters
        final_means = []
        final_centroids = []
        flg = False
        mn=0

        for loop in range(init_times):
            means = means_init(k,self.vectors)
            if mean_verbose==True:
                plot_means(means)
                plt.xlabel(str(self.level) + ' initial')
                plt.show()
            prev_means = means
            centroids = [0 for i in range(no_vectors)]
            for t in range(no_of_times):
                if mean_verbose==True:
                    plot_means(means)
                new_means=[[0 for j in range(vec_length)] for i in range(k)]

                # print(t,'th iteration:---------------------')
                # print(means)
                cnt = [0 for i in range(k)]
                for i in range(len(self.vectors)):
                    mn_dist = distance(means[0],self.vectors[i])
                    mn_idx = 0
                    # print(vectors[i],means[0],mn_dist)
                    for j in range(1,k):
                        dist = distance(means[j],self.vectors[i])
                        if dist<mn_dist:
                            mn_idx=j
                            mn_dist=dist
                        # print(vectors[i],means[j],dist,mn_idx,mn_dist)
                    centroids[i]=mn_idx
                    cnt[mn_idx]+=1
                    for j in range(vec_length):
                        new_means[centroids[i]][j]+=self.vectors[i][j]

                for i in range(k):
                    for j in range(vec_length):
                        if cnt[i]!=0:
                            new_means[i][j]/=cnt[i]
                means=new_means
                if prev_means==means:
                    break
                prev_means=means
            if mean_verbose==True:
                plt.xlabel('level: ' + str(self.level) + ' cluster: ' + str(self.cluster_no))
                plt.show()
            J = self.cost(means)
            if flg==False:
                mn=J
                final_means = means
                final_centroids=centroids
                flg=True
            else:
                if J<mn:
                    mn=J
                    final_means = means
                    final_centroids = centroids


        sub_clus = {}
        for i in range(no_vectors):
            sub_clus.setdefault(final_centroids[i],[]).append(self.vectors[i])

        for k,v in sub_clus.items():
            self.sub_clusters.append(cluster(v,final_means[k],self.level+1,k))
        self.no_of_sub_clusters = len(self.sub_clusters)
        if self.no_of_sub_clusters==1:
            self.sub_clusters[0].end=True
        print('Possible clusters: ',self.no_of_sub_clusters,'level: ',self.level)
        print('Vectors: ',self.no_vectors)
        for k in range(self.no_of_sub_clusters):
            print(self.sub_clusters[k].no_vectors)

    def K_Mean_Clusters(self):
        if self.end==False:
            self.K_Mean()
            for i in range(self.no_of_sub_clusters):
                if len(self.sub_clusters[i].vectors)<min_cluster_size :
                    self.sub_clusters[i].end = True
                cluster.K_Mean_Clusters(self.sub_clusters[i])

    def plot_cluster(self,recursive = True):
        for i in range(self.no_of_sub_clusters):
            for j in range(self.sub_clusters[i].no_vectors):
                plt.scatter(self.sub_clusters[i].vectors[j][0],self.sub_clusters[i].vectors[j][1],c=c[i])
        if self.no_of_sub_clusters!=0:
            plt.xlabel('level: ' + str(self.level) + ' cluster: ' + str(self.cluster_no))
            plt.show()
            if recursive==True:
                for i in range(self.no_of_sub_clusters):
                    if self.sub_clusters[i]!=None:
                        self.sub_clusters[i].plot_cluster()


#paramteres to set


filename = "vectors"

def readFile(filename):
    conv_type=list
    vectors = []
    with open(filename,'r') as f:
        for line in f:
            vectors.append(conv_type([float(i) for i in line.split()]))
    return vectors

vectors = readFile(file_vectors)


print('constructing hierarchical clustering...\n\n')
root = cluster(vectors,None,0)
root.K_Mean_Clusters()

print('hierarchical cluster has been constructed\n\n')

candid_vectors = readFile(file_candid_vectors)

print('starting searching procedure')

template = 'nearest vector: {0} distance: {1} candid vector: {2}\n'

total_time = 0

with open(output_file,'w') as f:
    for i in range(len(candid_vectors)):
        start = time.time()
        found_vec = root.search(candid_vectors[i])
        end = time.time()
        total_time += end-start
        f.write(template.format(str(found_vec[0]),str(distance(found_vec[0],candid_vectors[i])),str(candid_vectors[i])))

print('searching procedure has been ended.')
print('time taken by searching: '+str(total_time)+'s')
if plot_verbose==True:
    root.plot_cluster()


ans = input('\n\nwant to check accuracy by comparing it with brute - force solution[y\\N]:  ')


if ans.lower() == 'y':
    print('bruter-force searching has been begun')

    total_time = 0

    out_brute_file = '.ans_brute_force'

    with open(out_brute_file, 'w') as f:
        for i in range(len(candid_vectors)):
            start = time.time()
            found_vec = find_nearest(vectors, candid_vectors[i])
            end = time.time()
            total_time+=end-start
            f.write(template.format(str(found_vec[0]), str(distance(found_vec[0], candid_vectors[i])),str(candid_vectors[i])))

    print('brute_force searching has been done')
    print('time taken by brute-force searching: '+str(total_time)+'s')


    print('\n\ncomparing both files for differences')

    csf = open(output_file,'r')
    cs = csf.read().split('\n')
    csf.close()
    bsf = open(out_brute_file,'r')
    bs = bsf.read().split('\n')
    bsf.close()
    difference = 0

    for i in range(len(cs)):
        if cs[i]!=bs[i]:
            difference+=1

    print('total number of differences: ' + str(difference))
    print('Searching accuracy: ',str((1-(difference/len(cs)))*100) + '%')
