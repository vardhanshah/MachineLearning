import math,sys
#import matplotlib.pyplot as plt

def update(ls):
    mean=sum(ls)/float(len(ls))
    sigma=0
    for i in ls:
        sigma+=(mean-i)**2
    return mean,sigma/float(len(ls))

def calculate_ans(mean,sigma,val):
    return (1/math.sqrt(2*math.pi*sigma))*math.exp(math.pow(mean-val,2)*(-1)/(2*sigma))


"""--------------------Preparing for input-----------------------"""

taken=sys.argv[1:]
if len(taken)==0 :
    show="Give\nNeeded: (1)-Input File Path/Name\n"
    show+="Optional: (2)-No. of features needed for prediction\n"
    show+="Optional: (3)-Give file for prediction task otherwise it will use the input file\n"
    show+="for measure its accuracy\n\n"
    show+="Note: input should be separated by space( )\n\n"
    show+="Note: both input file should contain data in following manner:\n"
    show+="\t\t\tfeature1,feature2[,feature3,...],outcome\n\n\n"
    show+="and file input should be seperated by comma(,) \n"
    print(show)
    in_rep="Input in manner : (1) (2) (3)\n"
    taken=input(in_rep).split()

#taken=["iris.data","4"]
fname1=taken[0]
file = open(fname1,"r")
in_var = len(file.readline().split(','))-1
file.seek(0)
fname2=fname1
features=in_var
if len(taken)==3:
    fname2=taken[-1]
if len(taken)==2:
    features=min(int(taken[1]),in_var)
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
c=["red","blue","green"]

"""--------------------Zone Ended----------------------------"""




"""---------------------Taking Input-----------------------"""

for line in file:
    count+=1
    tmp=line.split(',')

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
        #print(ls[-1])
        color[ls[-1]]=c[k]
        k+=1

"""----------------------Zone Ended----------------------"""



"""---------------------Training Zone---------------------"""

for val in prob.values():
    val/=count

for key,val in types.items():
    for i in range(in_var) :
        x,y=update([x[i] for x in val])
        mean[key].append(x)
        sigma[key].append(y)

"""---------------------Zone Ended------------------------"""



"""--------------Prediction Zone started------------------"""

if fname2==fname1 :
    file.seek(0)
else:
    file = open(fname2,"r")
acc=0
total_input=0
for line in file:
    mx_ans=""
    mx=0
    ls=[ty(val) for ty,val in zip(in_type,line.split(','))]
    if ls[-1][-1]=='\n':
        ls[-1]= ls[-1][0:-1]
    for key in mean.keys():
        ans=1
        for i in range(features):
            ans*=calculate_ans(mean[key][i],sigma[key][i],ls[i])
        ans*=prob[key]
        if ans>mx:
            mx=ans
            mx_ans=key
    print("original: "+ls[-1],"prediction: "+mx_ans)
    if mx_ans==ls[-1]:
        acc+=1
    #plt.scatter(ls[0],ls[1],c=color[ls[-1]],marker="<")
    #plt.scatter(ls[0],ls[1],c=color[mx_ans],marker=">")
    total_input+=1

print("Accuracy: ",100*acc/total_input,"%",sep="")
#plt.show()

"""-----------------Zone Ended--------------------------"""

