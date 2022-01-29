import numpy as np

alg = 'MPVNN'  
ctype = 'OV'

basedir = '..\\Data\\'
outbasedir = "..\\Out\\"

topmax = 3
thrval1 = 0.85
thrval2 = thrval1


edges = []
pgenes = set()
with open(basedir+'pi3k-akt.txt', 'r') as f:
    for line in f:
        lineData = line.strip().split("\t")
        pgenes.add(lineData[0])
        pgenes.add(lineData[1])
        


        
count = 0
fpgenes = []
fogenes = []
with open(basedir+ctype+'_expr', 'r') as f:
    for line in f:
        count += 1
        lineData = line.rstrip().split("\t")
        if count == 1:
            continue
        if lineData[0] in pgenes: 
            fpgenes.append(lineData[0])
        else:
            fogenes.append(lineData[0])
            

fallgenes = fpgenes + fogenes
numgenes = len(fallgenes)


edgesp = []
with open(basedir+'pi3k-akt.txt', 'r') as f:
    for line in f:
        lineData = line.rstrip().split("\t")
        if lineData[0] not in fpgenes or lineData[1] not in fpgenes:
            continue
        edgesp.append(lineData[0]+"#"+lineData[1]) 

count = 0
with open(outbasedir+alg+'_'+ctype+'_results.tsv', 'r') as f:
    for line in f:
        count += 1
        lineData = line.rstrip().split("\t")
        if count == 1:
            continue
        elif count == 2:
            W1 = np.array(lineData).astype(np.float)[np.newaxis,:]
        elif count <= numgenes+1:
            W1 = np.concatenate((W1,np.array(lineData).astype(np.float)[np.newaxis,:]), axis=0)
        elif count == numgenes+2:
            W2 = np.array(lineData).astype(np.float)[np.newaxis,:]
        else:
            W2 = np.concatenate((W2,np.array(lineData).astype(np.float)[np.newaxis,:]), axis=0)


sindval = (-W2).argsort(axis=0)
topgenes = []
count = 0
for ind in sindval:
    topgenes.append(fallgenes[ind[0]])
    count += 1
    if count == topmax:
        break


print(ctype)

tW1 = W1.flatten()
tW1 = tW1[tW1 != 0]
thrmax1 = np.quantile(tW1,thrval1)
tW2 = W2.flatten()
tW2 = tW2[tW2 != 0]
thrmax2 = np.quantile(tW2,thrval2)
for index in range(topmax):
    print("\nTop Gene Set "+ str(index+1) + " : ")
    candgene = topgenes[index]
    topsgenes = [candgene]
    exclgenes = []
    while(1>0):
        maxindall = np.argsort(W1[:,fallgenes.index(candgene)])
        counter = -1
        maxind = maxindall[counter]
        valt = -2
        if candgene == topgenes[index]:
            valt = 0
        while (fallgenes[maxind] == candgene or fallgenes[maxind] == topsgenes[valt]):
            counter -= 1
            maxind = maxindall[counter]
        if W1[maxind,fallgenes.index(candgene)] < thrmax1:
            break
        if W1[maxind,maxind] < thrmax1:
            break
        if fallgenes[maxind] in topsgenes:
            topsgenes.append(fallgenes[maxind]+":back")
            break
        if W2[maxind,0] < thrmax2:
            break
        if candgene == topgenes[index]:
            exclgenes = [fallgenes[maxind]]
        candgene = fallgenes[maxind]
        topsgenes.append(candgene)
        
    
    candgene = topgenes[index]
    topsgenes.append("Other Path(top)")
    topsgenes.append(candgene)
    while(1>0):
        maxindall = np.argsort(W1[:,fallgenes.index(candgene)])
        counter = -1
        maxind = maxindall[counter]
        valt = -2
        omgenes = []
        if candgene == topgenes[index]:
            valt = 0
            omgenes = [genet for genet in exclgenes]
        while (fallgenes[maxind] == candgene or fallgenes[maxind] == topsgenes[valt] or fallgenes[maxind] in omgenes):
            counter -= 1
            maxind = maxindall[counter]
        if W1[maxind,fallgenes.index(candgene)] < thrmax1:
            break
        if W1[maxind,maxind] < thrmax1:
            break
        if fallgenes[maxind] in topsgenes:
            topsgenes.append(fallgenes[maxind]+":back")
            break
        if W2[maxind,0] < thrmax2:
            break
        candgene = fallgenes[maxind]
        topsgenes.append(candgene)


    for i in range(len(topsgenes)):
        if i == 0:
            printgene = "(top)"+topsgenes[i]
        else:
            printgene = topsgenes[i]
        print(printgene+",",end = '')
    
    









