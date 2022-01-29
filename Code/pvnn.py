from lifelines.utils import concordance_index
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l1, l2
#from keras.optimizers import adam_v2, gradient_descent_v2
from keras.optimizers import Adam, SGD
import tensorflow as tf
import keras.backend as K
from tensorflow.python.framework import ops
basedir = "..\\Data\\"
outbasedir = "..\\Out\\"


alg = 'PVNN'
ctype = 'BLCA'

dataseed = 100
algnpseed = 13300000
algtfseed = 133000000

survindex = 4 #DSS
setnum = 300
setrunnum = 5
runnum = 20


activationlist = ['tanh','sigmoid','relu']
#optimizerlist = [adam_v2.Adam,gradient_descent_v2.SGD]
optimizerlist = [Adam,SGD]
learningratelist = [0.1,0.01,0.001]
batchsizelist = [16,32,64,128]
epochlist = [10,25,50,100]
regularizerlist=[l1,l2]
regparamlist = [1.0,1e-1,1e-2,1e-3,1e-5,0.0]



class CustomConnected(Dense):

    def __init__(self,units,connections,**kwargs):

        self.connections = connections                        

        super(CustomConnected,self).__init__(units,**kwargs)  


    def call(self, inputs):
        output = K.dot(inputs, self.kernel * self.connections)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output


def cibound_loss(y_true, y_pred):
    if (not tf.executing_eagerly()):
        print("Eager execution needed for loss")
    y_event_times  = tf.gather(y_true,0,axis=1)
    y_event_observed = tf.gather(y_true,1,axis=1)
    y_pred_exp = tf.exp(y_pred)
    y_pred_nexp = tf.exp(-y_pred)
    y_pred_exp_unstacked = tf.unstack(y_pred_exp)
    y_event_times_unstacked = tf.unstack(y_event_times)
    processed = []
    for texp,ttime in zip(y_pred_exp_unstacked,y_event_times_unstacked):
        event_time_maskedo = tf.cast(tf.greater(y_event_times,ttime),tf.float32)
        event_time_masked = tf.expand_dims(event_time_maskedo,axis=1)
        indloss = tf.subtract(tf.multiply(tf.multiply(y_pred_nexp,event_time_masked),texp),event_time_masked)
        indlosssum = tf.reduce_sum(indloss,axis=0)
        processed.append(indlosssum)
    allloss = tf.concat(processed, 0)
    allloss_masked = tf.multiply(allloss,y_event_observed)
    loss = tf.reduce_sum(allloss_masked,axis=0)
    return loss



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
tf.random.set_seed(algtfseed)  



edgesp = []
with open(basedir+'pi3k-akt.txt', 'r') as f:
    for line in f:
        lineData = line.rstrip().split("\t")
        if lineData[0] not in fpgenes or lineData[1] not in fpgenes:
            continue
        edgesp.append(lineData[0]+"#"+lineData[1]) 

W = np.identity(numnodes)
for edge in edgesp:
    ab = edge.split("#")
    W[fallgenes.index(ab[0]),fallgenes.index(ab[1])] = 1
    W[fallgenes.index(ab[1]),fallgenes.index(ab[0])] = 1



valcimax = -1.0
for j in range(setnum):
    #print("Setting : " + str(j))        

    hactivationval = np.random.choice(activationlist)
    oactivationval = np.random.choice(activationlist)
    optimizerval = np.random.choice(optimizerlist)
    learningrateval = np.random.choice(learningratelist)
    batchsizeval = np.random.choice(batchsizelist)
    epochsval = np.random.choice(epochlist)
    hkregularizerval = np.random.choice(regularizerlist)
    hkregparamval = np.random.choice(regparamlist)
    okregularizerval = np.random.choice(regularizerlist)
    okregparamval = np.random.choice(regparamlist)

    valciall = []
    for train_index1, test_index1 in skf2.split(X_trainall,event_observed_trainall):
        X_train, X_vd = X_trainall[train_index1], X_trainall[test_index1]
        event_times_train, event_times_vd = event_times_trainall[train_index1], event_times_trainall[test_index1]
        event_observed_train, event_observed_vd = event_observed_trainall[train_index1], event_observed_trainall[test_index1]
        
        for i in range(setrunnum):
            model = Sequential()
            model.add(CustomConnected(numnodes, W, input_dim=numnodes, activation=hactivationval,
                                  kernel_regularizer=hkregularizerval(hkregparamval)))                                  
            model.add(Dense(1, activation=oactivationval,
                          kernel_regularizer=okregularizerval(okregparamval)))
            opt = optimizerval(learning_rate=learningrateval) 
            model.compile(loss=cibound_loss,optimizer=opt,run_eagerly=True)
            history = model.fit(X_train, np.concatenate((event_times_train,event_observed_train),axis=1),
                                batch_size=batchsizeval,epochs=epochsval,verbose=0)
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
        hactivationmax = hactivationval
        oactivationmax = oactivationval
        optimizermax = optimizerval
        learningratemax = learningrateval
        batchsizemax = batchsizeval
        epochsmax = epochsval
        hkregularizermax = hkregularizerval
        hkregparammax = hkregparamval
        okregularizermax = okregularizerval
        okregparammax = okregparamval    

    

cimetrics = []

allweights1 = np.zeros((numnodes,numnodes))
allweights2 = np.zeros((numnodes,1))

for i in range(runnum):
    model = Sequential()
    model.add(CustomConnected(numnodes, W, input_dim=numnodes, activation=hactivationmax,
                                  kernel_regularizer=hkregularizermax(hkregparammax)))                                  
    model.add(Dense(1, activation=oactivationmax,
                          kernel_regularizer=okregularizermax(okregparammax)))
    opt = optimizermax(learning_rate=learningratemax) 
    model.compile(loss=cibound_loss,optimizer=opt,run_eagerly=True)        
    history = model.fit(X_trainall, np.concatenate((event_times_trainall,event_observed_trainall),axis=1),
                            batch_size=batchsizemax,epochs=epochsmax,verbose=0)
    y_pred = model.predict(X_test)
    if np.isnan(np.sum(np.array(y_pred))):
        cimetrics.append(0.5)
    else:
        cimetrics.append(concordance_index(event_times_test, y_pred, event_observed_test))
    
    weights = model.get_weights()
    allweights1 += np.abs(weights[0]*W)
    allweights2 += np.abs(weights[2])
        
    del model
    K.clear_session()
    ops.reset_default_graph()               
        


print(alg)    
print(ctype)
print(np.mean(cimetrics))
print(np.std(cimetrics))


allweights1 = allweights1/runnum
allweights2 = allweights2/runnum

outfilename = outbasedir + alg + '_' + ctype + '_results.tsv' 
with open(outfilename, 'w') as f:
    for i in range(len(cimetrics)):
        val = "\t"
        if i == len(cimetrics) - 1:
            val = "\n"
        f.write(str(cimetrics[i])+val)
    for i in range(allweights1.shape[0]):
        for j in range(allweights1.shape[1]):
            val = "\t"
            if j == allweights1.shape[1] - 1:
                val = "\n"
            f.write(str(allweights1[i,j])+val)
    for i in range(allweights2.shape[0]):
        f.write(str(allweights2[i,0])+"\n")
        
        





