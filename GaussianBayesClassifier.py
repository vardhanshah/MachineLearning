import math
import matplotlib.pyplot as plt
def update(ls):
    mean=sum(ls)/float(len(ls))
    sigma=0
    for i in ls:
        sigma+=(mean-i)**2
    return mean,sigma/float(len(ls))

def calculate_ans(mean,sigma,val):
    return (1/math.sqrt(2*math.pi*sigma))*math.exp(math.pow(mean-val,2)*(-1)/(2*sigma))

in_rep="Give -> Input File Path/Name, no. of features in data set, no. of features needed for prediction: "
file_name,in_var,features=[ty(val) for ty,val in zip([str,int,int],input(in_rep).split())]
in_var=int(in_var)
features=int(features)
file = open(file_name,"r")
in_type=[]
for i in range(in_var):
    in_type.append(float)
in_type.append(str)
types={}
mean={}
sigma={}
prob={}
count=0
color={}
k=0
c=["red","blue","green"]
complete_input=[]
for line in file:

    count+=1
    ls=[ty(val) for ty,val in zip(in_type,line.split(','))]
    if ls[-1][-1]=='\n':
        ls[-1]= ls[-1][0:-1]
    if ls[-1] in types:
        types[ls[-1]].append(ls[0:-1])
        prob[ls[-1]]+=1
    else :
        types[ls[-1]]=[ls[0:-1]]
        prob[ls[-1]]=1
        mean[ls[-1]]=[]
        sigma[ls[-1]]=[]
        #print(ls[-1],k)
        color[ls[-1]]=c[k]
        k+=1
    complete_input.append(ls)
for val in prob.values():
    val/=count

for key,val in types.items():
    for i in range(in_var) :
        x,y=update([x[i] for x in val])
        mean[key].append(x)
        sigma[key].append(y)
file.close()
acc=0

for ls in complete_input:
    mx_ans=""
    mx=0
    for key in mean.keys():
        ans=1
        for i in range(features):
            ans*=calculate_ans(mean[key][i],sigma[key][i],ls[i])
        ans*=prob[key]
        if ans>mx:
            mx=ans
            mx_ans=key
    if mx_ans==ls[-1]:
        acc+=1
    plt.scatter(ls[0],ls[1],c=color[ls[-1]],marker="<")
    plt.scatter(ls[0],ls[1],c=color[mx_ans],marker=">")


print("Accuracy: ",100*acc/len(complete_input),"%",sep="")
#plt.xlabel("sepal-length")
#plt.ylabel("sepal-width")
plt.show()
