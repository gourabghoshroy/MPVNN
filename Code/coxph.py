from lifelines import CoxPHFitter
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings
basedir = "..\\Data\\"
outbasedir = "..\\Out\\"


alg = 'COXPH'
ctype = 'BLCA'


dataseed = 100
algnpseed = 13333300


survindex = 4 #DSS
setnum = 10
setrunnum = 5
runnum = 20


penalizerlist = [0.0,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1.0]
l1ratiolist = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]




pgenes = set()
with open(basedir+'pi3k-akt.txt', 'r') as f:
    for line in f:
        lineData = line.strip().split("\t")
        pgenes.add(lineData[0])
        pgenes.add(lineData[1])


        
count = 0
countp = 0
counto = 0
samples = []
fpgenes = []
fogenes = []
with open(basedir+ctype+'_expr', 'r') as f:
    for line in f:
        count += 1
        lineData = line.rstrip().split("\t")
        if count == 1:
            colc = 0
            for val in lineData:
                colc += 1
                if colc == 1:
                    continue
                samples.append(val)
            continue
        if lineData[0] in pgenes: 
            countp += 1
            fpgenes.append(lineData[0])
            Xpt = np.zeros((1,colc-1))
            colc1 = 0
            for val in lineData:
                colc1 += 1
                if colc1 == 1:
                    continue
                Xpt[0,colc1-2] = val
            if countp == 1:
                Xp = Xpt    
            else:
                Xp = np.concatenate((Xp,Xpt), axis=0)
        else:
            counto += 1
            fogenes.append(lineData[0])
            Xot = np.zeros((1,colc-1))
            colc1 = 0
            for val in lineData:
                colc1 += 1
                if colc1 == 1:
                    continue
                Xot[0,colc1-2] = val
            if counto == 1:
                Xo = Xot   
            else:
                Xo = np.concatenate((Xo,Xot), axis=0)
            

fallgenes = fpgenes + fogenes           
Xexp =  np.concatenate((Xp,Xo), axis=0)               
Xexp = np.transpose(Xexp)

count = 0
sursamples = []
surpatients = []
event_times = []
event_observed = []
mcount = 0
with open(basedir+ctype+'_survival.txt', 'r') as f:
    for line in f:
        count += 1
        if count == 1:
            continue
        lineData = line.rstrip().split("\t")
        valt = int(lineData[0].split("-")[-1])
        if lineData[0].split("-")[0] == "TCGA" and valt >= 10 and valt <= 14:
            continue
        if lineData[0] in samples and lineData[1] not in surpatients and lineData[0] not in sursamples:
            if len(lineData) < survindex+2 or len(lineData[survindex]) == 0 or len(lineData[survindex+1]) == 0:
                continue
            if len(lineData) >= 11 and lineData[10] == "Redacted":
                continue
            sursamples.append(lineData[0])
            surpatients.append(lineData[1])
            event_times.append(float(lineData[survindex+1]))
            event_observed.append(float(lineData[survindex]))
            if mcount == 0:
                X = Xexp[samples.index(lineData[0]),:][np.newaxis,:]
            else:
                X = np.concatenate((X,Xexp[samples.index(lineData[0]),:][np.newaxis,:]), axis=0)
            mcount += 1


event_times = np.array(event_times)[:,np.newaxis]
event_observed = np.array(event_observed)[:,np.newaxis]
numnodes = X.shape[1]


np.random.seed(dataseed)
all_index = np.random.permutation(X.shape[0])
X = X[all_index]
event_times = event_times[all_index]
event_observed = event_observed[all_index]


skf1 = StratifiedKFold(n_splits=5)
skf2 = StratifiedKFold(n_splits=4)

for train_index, test_index in skf1.split(X,event_observed):
    X_trainall, X_test = X[train_index], X[test_index]
    event_times_trainall, event_times_test = event_times[train_index], event_times[test_index]
    event_observed_trainall, event_observed_test = event_observed[train_index], event_observed[test_index]
    break

scaler = StandardScaler().fit(X_trainall)
X_trainall = scaler.transform(X_trainall) 
X_test = scaler.transform(X_test)  


# Algorithm
np.random.seed(algnpseed)

column_values = ['times','observed']
cov_values = []
for i in range(numnodes):
    val1 = "gene" + str(i+1)
    cov_values.append(val1)
column_values = column_values + cov_values




valcimax = -1.0
for j in range(setnum):
    #print("Setting : " + str(j)) 
    penalizerval = np.random.choice(penalizerlist)
    l1ratioval = np.random.choice(l1ratiolist)
    
    valciall = []
    for train_index1, test_index1 in skf2.split(X_trainall,event_observed_trainall):
        X_train, X_vd = X_trainall[train_index1], X_trainall[test_index1]
        event_times_train, event_times_vd = event_times_trainall[train_index1], event_times_trainall[test_index1]
        event_observed_train, event_observed_vd = event_observed_trainall[train_index1], event_observed_trainall[test_index1]
        
        arr1 = np.concatenate((event_times_train,event_observed_train,X_train),axis=1)
        arr2 = np.concatenate((event_times_vd,event_observed_vd,X_vd),axis=1)
    
        df1 = pd.DataFrame(data = arr1, columns = column_values)
        df2 = pd.DataFrame(data = arr2, columns = column_values)
        
        warnings.filterwarnings("error")
        cph = CoxPHFitter(penalizer=penalizerval,l1_ratio=l1ratioval)
        try:
            ipt = 0.01*np.random.randn(len(df1.columns)-2,1)
            ipt = np.ravel(ipt)
            cph.fit(df1,'times','observed',initial_point=ipt)
        except Exception as e:
            s = str(e).split(" have very low variance")
            if len(s) > 1:
                if "[" in s[0] and "]" in s[0]:
                    testval = s[0].split("[")[-1].split("]")[0]
                    excol = testval.replace("'","").split(", ")
                else:
                    testval = s[0].split(" ")[-1]
                    excol = [testval]
                df1.drop(excol, axis = 1, inplace = True)
                df2.drop(excol, axis = 1, inplace = True)
        del cph
        warnings.filterwarnings("default")
        for i in range(setrunnum):
            cph = CoxPHFitter(penalizer=penalizerval,l1_ratio=l1ratioval)
            try:
                ipt = 0.01*np.random.randn(len(df1.columns)-2,1)
                ipt = np.ravel(ipt)
                cph.fit(df1,'times','observed',initial_point=ipt)
                valciall.append(cph.score(df2,'concordance_index'))
            except:
                valciall.append(0.5)            
            del cph
    
    valci = np.mean(valciall)
    if valci > valcimax:
        valcimax = valci
        penalizermax = penalizerval
        l1ratiomax = l1ratioval
        


arr1 = np.concatenate((event_times_trainall,event_observed_trainall,X_trainall),axis=1)
arr2 = np.concatenate((event_times_test,event_observed_test,X_test),axis=1)
    
df1 = pd.DataFrame(data = arr1, columns = column_values)
df2 = pd.DataFrame(data = arr2, columns = column_values)

warnings.filterwarnings("error")
cph = CoxPHFitter(penalizer=penalizermax,l1_ratio=l1ratiomax)
try:
    ipt = 0.01*np.random.randn(len(df1.columns)-2,1)
    ipt = np.ravel(ipt)
    cph.fit(df1,'times','observed',initial_point=ipt)
except Exception as e:
    s = str(e).split(" have very low variance")
    if len(s) > 1:
        if "[" in s[0] and "]" in s[0]:
            testval = s[0].split("[")[-1].split("]")[0]
            excol = testval.replace("'","").split(", ")
        else:
            testval = s[0].split(" ")[-1]
            excol = [testval]
        df1.drop(excol, axis = 1, inplace = True)
        df2.drop(excol, axis = 1, inplace = True)
del cph
warnings.filterwarnings("default")
cimetrics  = []

for i in range(runnum):
    cph = CoxPHFitter(penalizer=penalizermax,l1_ratio=l1ratiomax)
    try:
        ipt = 0.01*np.random.randn(len(df1.columns)-2,1)
        ipt = np.ravel(ipt)
        cph.fit(df1,'times','observed',initial_point=ipt)
        cimetrics.append(cph.score(df2,'concordance_index'))
    except:
        cimetrics.append(0.5)
    del cph
        
  
print(alg)
print(ctype)
print(np.mean(cimetrics))
print(np.std(cimetrics))

outfilename = outbasedir + alg + '_' + ctype + '_results.tsv' 
with open(outfilename, 'w') as f:
    for i in range(len(cimetrics)):
        val = "\t"
        if i == len(cimetrics) - 1:
            val = "\n"
        f.write(str(cimetrics[i])+val)
        
        
