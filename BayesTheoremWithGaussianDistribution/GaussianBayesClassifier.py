import math,sys
import matplotlib.pyplot as plt
import matplotlib.colors
c=[]
for name in matplotlib.colors.cnames.keys():
    c.append(name)
def update(ls):
    mean=sum(ls)/float(len(ls))
    sigma=0
    for i in ls:
        sigma+=(mean-i)**2
    return mean,sigma/float(len(ls))

def calculate_ans(mean,sigma,val):
    return (1/math.sqrt(2*math.pi*sigma))*math.exp(math.pow(mean-val,2)*(-1)/(2*sigma))


"""--------------------Preparing for input-----------------------"""
#taken=["dataset1.csv"]

taken=sys.argv[1:]

if len(taken)==0 :
    taken=input().split()

#taken=["iris.data","4"]
fname1=taken[0]
file = open(fname1,"r")
in_var = len(file.readline().split(','))-1
file.seek(0)
fname2=fname1
features=in_var
if len(taken)==3:
    if taken[1].isnumeric():
        features=min(int(taken[1]),in_var)
        fname2=taken[2]
    else :
        features=min(int(taken[2]),in_var)
        fname2=taken[1]

if len(taken)==2:
    if taken[1].isnumeric():
        features=min(int(taken[1]),in_var)
    else:
        fname2=taken[-1]

in_type=[]
for i in range(in_var):
    in_type.append(float)
in_type.append(str)

"""--------------------Zone ended--------------------------"""




"""--------------------Declaration Zone---------------------"""

types={}
mean={}
sigma={}
prob={}
count=0
color={}
k=0

"""--------------------Zone Ended----------------------------"""




"""---------------------Taking Input-----------------------"""

for line in file:
    count+=1
    tmp=line.split(',')

    ls=[ty(val) for ty,val in zip(in_type,line.split(','))]
    if ls[-1][-1]=='\n' or ls[-1][-1]==' ':
        ls[-1]= ls[-1][0:-1]
    if ls[-1] in types:
        types[ls[-1]].append(ls[0:-1])
        prob[ls[-1]]+=1
    else :
        types[ls[-1]]=[ls[0:-1]]
        prob[ls[-1]]=1
        mean[ls[-1]]=[]
        sigma[ls[-1]]=[]
        #print(ls[-1])
        color[ls[-1]]=c[k]
        k+=1
    #print(ls)

"""----------------------Zone Ended----------------------"""



"""---------------------Training Zone---------------------"""

for key in prob.keys():
    prob[key]=prob[key]/count
for key,val in types.items():
    for i in range(in_var) :
        x,y=update([x[i] for x in val])
        mean[key].append(x)
        sigma[key].append(y)
#print(types)

"""---------------------Zone Ended------------------------"""



"""--------------Prediction Zone started------------------"""

if fname2==fname1 :
    file.seek(0)
else:
    file = open(fname2,"r")
acc=0
total_input=0
for line in file:
    if total_input>5000:
        break
    mx_ans=""
    mx=0
    ls=[ty(val) for ty,val in zip(in_type,line.split(','))]
    if ls[-1][-1]=='\n' or ls[-1][-1]==' ':
        ls[-1]= ls[-1][0:-1]
    for key in mean.keys():
        ans=1
        for i in range(features):
            if sigma[key][i]!=0:
                ans*=calculate_ans(mean[key][i],sigma[key][i],ls[i])
        ans*=prob[key]
        if ans>mx:
            mx=ans
            mx_ans=key

    if mx_ans=='':
        for key in types.keys():
            mx_ans=key
            break
    print("original: ",ls[-1],"prediction: ",mx_ans)
    if mx_ans==ls[-1]:
        acc+=1

    plt.scatter(ls[0],ls[1],c=color[ls[-1]],marker="<")
    plt.scatter(ls[0],ls[1],c=color[mx_ans],marker=">")
    total_input+=1

print("Accuracy: ",100*acc/total_input,"%",sep="")
plt.show()

"""-----------------Zone Ended--------------------------"""
