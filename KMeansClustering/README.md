<pre>

Here, I have implemented hierarchical clustering, basically tree of clusters.
Every node can have maximum number of sub-clusters defined by programmer in config file. (more on config file below !)

Input-Data-Set needed to be vectors, vectors of d-Dimension, where d>0 (positive number).

After the hierarchical clustering has been constructed, one of the uses of it can be for searching the neareset vector for a candid-vector.
(Neareset-vector is decided by euclidian-distance.)

candid-vector needed to have dimension d, same as input-vectors.

So I used this hierarchical clustering for searching, In code execuction, you can also check seaching accuracy, by comparing it with brute-force-searching accuracy.


Clustering is done with help of K-Means algorithm. Intial values of k-means clustering has been choosen with help of K-means++ algorithm.

Implementation is written in python3

matplotlib is required (if you want to plot the cluster or mean path)

You need to define hyper-parameters of algorithm in config file.
config file should be located in current directory.

In config file,

all parameter inside config file should preserve below order,

	Parameters:
		Input_vectors_file
		Input_candid_vectors_file
		Output_file
		Minimum_cluster_size
		number_of_sub_clusters
		mean plot verbose
		cluster_plot_verbose
		coverge_loop_times
		optimization_loop_times

No specific name of the parameters is required, but order should be preserved


In above parameters, 

	Input_vectors_file denotes, file in which number of input vectors given

	Input_candid_vectors_file denotes, file in which number of candid-vectors defined

	Output_file denotes, file in which output of search result will be written

	Minimum_cluster_size denotes, minimum cluster size needed for further clustering of a node, set the value of it an integer

	number_of_sub_clusters denotes, Maximum possible sub-clusters will occur for each node, set the value of it an integer

	mean_plot_verbose denotes, plotting of inital means choosen and path they take to converge, set the value of it true/false

	cluster_plot_verbose denotes, plotting of clusters for each node, set the value of it true/false
	
	Note: set mean_plot_verbose true, only when data size of input is small (because the plotting will occur for each 
		node, so for big size-data, number of nodes can be high) and dimension of input-vector >= 2
	
	Note: set cluster_plot_verbose true, only for data size of input is not so big, cause as the number of inputs increase.
		plotting time incereases also and plotting will occur for each sub-cluster
	
	converger_loop_times denotes, maximum number of iterations given to mean to converge a value
	
	Now, first mean in K-means++ initialization algorithms is decided randomly, so there is chance of getting at local-optimum in 
	clustering procedure, getting kind of bad cluster (now this has low impact for searching procedure). but it can be avoided by, one 
	of the methods of optimization, which is intializing mean and finding stable mean, repeat this process number of times.
	
	optimization_loop_times denotes above procedure.

	Note: if the number of optimization_loop_times increases by k, then K-Means hierarchical clustering construction is also 
		increased by k of original time.  

Example Config file:

	input_file_vectors = vectors
	input_file_candid_vectors = candid_vector
	output_file = ans_cluster
	minimum cluster size = 250
	number of sub clusters = 2
	mean plot verbose = false
	cluster plot verbose = false
	converge loop times = 1000
	optimization loop times = 1


with this paramters,

	Tree of cluster will be created, where each node will have two children, which means, each cluster will have two sub-clusters
	sub-cluster will be created until cluster-size>=250 

To execute:

	python3 cluster_tree_constructor.py

output:

	constructing hierarchical clustering...


	Possible clusters:  2 level:  0
	Vectors:  1000
	502
	498
	Possible clusters:  2 level:  1
	Vectors:  502
	221
	281
	Possible clusters:  2 level:  2
	Vectors:  281
	166
	115
	Possible clusters:  2 level:  1
	Vectors:  498
	218
	280
	Possible clusters:  2 level:  2
	Vectors:  280
	142
	138
	hierarchical cluster has been constructed


	starting searching procedure
	searching procedure has been ended.
	time taken by searching: 0.2417609691619873s


	want to check accuracy by comparing it with brute - force solution[y\N]:  y
	bruter-force searching has been begun
	brute_force searching has been done
	time taken by brute-force searching: 1.2946381568908691s


	comparing both files for differences
	total number of differences: 43

</pre>
