from lifelines.utils import concordance_index
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
#from keras.optimizers import adam_v2
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
from tensorflow.python.framework import ops
basedir = "..\\Data\\"
outbasedir = "..\\Out\\"


alg = 'CoxPASNet1P'
ctype = 'BLCA'

dataseed = 100
algnpseed = 16000000
algtfseed = 160000000

survindex = 4 #DSS
setnum = 30
setrunnum = 5
runnum = 20


#optimizerv = adam_v2.Adam
optimizerv = Adam
learningratelist = [1.0,0.5,0.1,0.05,0.025,0.01,0.0075,0.005,0.001,0.0001,0.00001]
epochlist = [10,25,50,100]
regparamlist = [1.0,0.5,0.1,0.05,0.01,0.005,0.001,0.00001]



def likelihood_loss(y_true, y_pred):
    if (not tf.executing_eagerly()):
        print("Eager execution needed for loss")
    y_event_times = tf.gather(y_true,0,axis=1)
    y_event_observed = tf.gather(y_true,1,axis=1)
    y_pred_exp = tf.exp(y_pred)
    y_event_times_unstacked = tf.unstack(y_event_times)
    processed = []
    for ttime in y_event_times_unstacked:
        event_time_maskedo = tf.cast(tf.greater_equal(y_event_times,ttime),tf.float32)
        event_time_masked = tf.expand_dims(event_time_maskedo,axis=1) 
        processed.append(tf.reduce_sum(tf.multiply(y_pred_exp,event_time_masked),axis=0)) 
    indloss = tf.multiply(tf.subtract(tf.math.log(tf.concat(processed, 0)),y_pred),y_event_observed)
    loss = tf.divide(tf.reduce_sum(indloss,axis=0),tf.reduce_sum(y_event_observed,axis=0))
    return loss





pgenes = set()
with open(basedir+'pi3k-akt.txt', 'r') as f:
    for line in f:
        lineData = line.strip().split("\t")
        pgenes.add(lineData[0])
        pgenes.add(lineData[1])


        
count = 0
countp = 0
samples = []
fpgenes = []
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
            Xexpt = np.zeros((1,colc-1))
            colc1 = 0
            for val in lineData:
                colc1 += 1
                if colc1 == 1:
                    continue
                Xexpt[0,colc1-2] = val
            if countp == 1:
                Xexp = Xexpt    
            else:
                Xexp = np.concatenate((Xexp,Xexpt), axis=0)
            


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
tf.random.set_seed(algtfseed) 


valcimax = -1.0
for j in range(setnum):
    #print("Setting : " + str(j))   
    learningrateval = np.random.choice(learningratelist)
    epochsval = np.random.choice(epochlist)
    regparamval = np.random.choice(regparamlist)

    valciall = []
    for train_index1, test_index1 in skf2.split(X_trainall,event_observed_trainall):
        X_train, X_vd = X_trainall[train_index1], X_trainall[test_index1]
        event_times_train, event_times_vd = event_times_trainall[train_index1], event_times_trainall[test_index1]
        event_observed_train, event_observed_vd = event_observed_trainall[train_index1], event_observed_trainall[test_index1]
        
        for i in range(setrunnum):
            model = Sequential()
            model.add(Dense(1, input_dim=numnodes, activation='tanh',
                            kernel_regularizer=l2(regparamval)))                                  
            model.add(Dense(1, kernel_regularizer=l2(regparamval), use_bias=False))
            opt = optimizerv(learning_rate=learningrateval)
            model.compile(loss=likelihood_loss,optimizer=opt,run_eagerly=True)
            history = model.fit(X_train, np.concatenate((event_times_train,event_observed_train),axis=1),
                                batch_size=X_train.shape[0],epochs=epochsval,verbose=0)
            y_vd = model.predict(X_vd)
            if np.isnan(np.sum(np.array(y_vd))):
                valciall.append(0.5)
            else:
                valciall.append(concordance_index(event_times_vd, y_vd, event_observed_vd))
            del model
            K.clear_session()
            ops.reset_default_graph() 
    
    
    valci = np.mean(valciall)
    if valci > valcimax:
        valcimax = valci
        learningratemax = learningrateval
        epochsmax = epochsval
        regparammax = regparamval

    

cimetrics = []



for i in range(runnum):
    model = Sequential()
    model.add(Dense(1, input_dim=numnodes, activation='tanh',
                            kernel_regularizer=l2(regparammax)))                                  
    model.add(Dense(1, kernel_regularizer=l2(regparammax), use_bias=False))
    opt = optimizerv(learning_rate=learningratemax)
    model.compile(loss=likelihood_loss,optimizer=opt,run_eagerly=True)      
    history = model.fit(X_trainall, np.concatenate((event_times_trainall,event_observed_trainall),axis=1),
                            batch_size=X_trainall.shape[0],epochs=epochsmax,verbose=0)
    y_pred = model.predict(X_test)
    if np.isnan(np.sum(np.array(y_pred))):
        cimetrics.append(0.5)
    else:
        cimetrics.append(concordance_index(event_times_test, y_pred, event_observed_test))
        
    del model
    K.clear_session()
    ops.reset_default_graph()               
        


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
