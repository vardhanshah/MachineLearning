import csv
import numpy as np
import os
import sys
verbose=True
if len(sys.argv)>1:
    verbose = True if sys.argv[1].title()=='True' else False

filename = os.path.join(os.getcwd(),'config')

#config file required to preserve the order of inputs
#all the parameter's value assignment should occur with '=' assignment operator

#config dictionary will contain all the parameters and values
config = {}
#params list will contain all the parameters info
params = []
with open(filename,'r') as file:
    for line in file:
        if line=='\n' or line=='' or line[0]=='#' :
            continue
        k,v = line.split('=')
        rm = ['\n','\t',' ']
        for i in rm:
            k = k.strip(i)
            v = v.strip(i)
        params.append(k)
        config[k] = v
print('Parameters Detected: ')
for i in params:
    print('\t',i)
#give the column range required
def cut(str):
    if '-' in str:
        s,e = [int(i) for i in str.split('-')]
        return [s-1,e]
    return [int(str)-1]

#takes the input from both files
def take_input(filename,header,columns,conv_type):
    data = []
    with open(filename,'r') as csvfile:
        csvreader = csv.reader(csvfile)
        if header.lower()=='true':
            csvreader.__next__()
        for row in csvreader:
            if len(columns)==1:
                data.append(conv_type(row[columns[0]]))
            else:
                data.append([conv_type(row[i]) for i in range(columns[0],columns[1])])
    return data
#in_columns stores the required columns intake of features
in_columns = cut(config[params[2]])
#out_columns stores the required columns intake of output
out_columns = cut(config[params[5]])

conv_type = float
#train_x stores input from input_filepath param.
train_x = np.array(take_input(config[params[0]],config[params[1]],in_columns,conv_type))
#train_y stores output from output_filepath param.
if len(params)==14 and config[params[13]].lower()=='true':
    conv_type=str
train_y = np.array(take_input(config[params[3]],config[params[4]],out_columns,conv_type))


#if one_hot_encoding required of output labels
def one_hot_encoding(data):
    cnt=0
    mapping={}
    for i in data:
        if i not in mapping.keys():
            mapping[i]=cnt
            cnt+=1
    data = np.array([np.int64(mapping[i]) for i in data],dtype=np.int64)
    num_classes = len(mapping)
    ext = np.zeros((len(data),int(num_classes)))
    for i in range(len(data)):
        ext[i,int(data[i])]=1.
    return ext



ptype = 0

if len(params)>=13:
    if config[params[12]].lower()=='true':
        if len(params)==14 and config[params[13]].lower()=='true':
            train_y = one_hot_encoding(train_y)
        ptype = 1


#sets value for number of nodes given in each layer
if ',' in config[params[6]]:
    nodes = [int(i) for i in config[params[6]].split(',')]
elif config[params[6]]!='':
    nodes = [int(config[params[6]])]
else :
    nodes = []

num_features = train_x.shape[1]



#all the required parameters for forward propagation to neural network
#fetching the main details of config file


input_size=len(train_y)
num_classes = train_y.shape[1]
bias_include = bool(config[params[7]].title())
learning_rate= float(config[params[8]])
epochs = int(config[params[9]])
mini_batches = min(int(config[params[10]]),input_size)
activation_dict = {'sigmoid':0,'tanh':1,'relu':2}
config[params[11]]=config[params[11]].split(',')
nodes.insert(0,num_features)
nodes.append(num_classes)
num_layers = len(nodes)
num_hidden_layers = len(nodes) - 2
if len(config[params[11]]) > num_hidden_layers:
    config[params[11]]=config[params[11]][:num_hidden_layers]
while len(config[params[11]])< num_hidden_layers:
    config[params[11]].append(config[params[11]][-1])
config[params[11]] = [i.lower().strip(' ') for i in config[params[11]]]
activation = [ activation_dict[func_name] for func_name in config[params[11]] ]
func_name = config[params[11]]


#print the final eqution
def print_equation(file):
    with open(file,'w') as f:
        show = "x denotes input\n\n"

        show += "y denotes output\n\n"

        for i in range(1,num_layers):
            show += "w"+str(i) + ', '
        show += "denotes weights\n\n"
        if bias_include:
            for i in range(1,num_layers):
                show+='b'+str(i)+', '
            show+='denotes biases\n\n'

        input_var = 'x'
        for i in range(1,num_layers-1):
            eq = 'w'+str(i) + '*'+input_var
            if bias_include:
                eq+=' + b'+str(i)
            acfunc = 'a' + str(i) + ' = ' +  func_name[i-1] + '('+eq + ')'
            show += acfunc +'\n\n'
            input_var='a'+str(i)
        no = num_layers-2
        if ptype==0:
            y = 'y = ' + 'relu' + '(' + 'w'+str(no+1)+'*a'+str(no)
        else:
            y = 'y = ' + 'softmax' + '(' + 'w'+str(no+1)+'*a'+str(no)
        if bias_include:
            y+=' + b'+str(no+1)
        y+=')'
        show += y+'\n'
        f.write(show)
    print("equations are stored in",file)

