import sys
import numpy as np
UNKNOWN_WORD = "<unk>"
binFN = sys.argv[1] #embedding file


directory = "twoTimesEmbed/"
embedding = directory + binFN

embedding2 = directory+ sys.argv[2] # second embedding file
# for window size 1. Each two words a, b, to get the #(a, b)/#(a)
embeddingNoNeg = directory + sys.argv[3]
embeddingNoNeg2 = directory+ sys.argv[4]

dictWords = {}
"""word embedding loading and vocabulary"""
wordCount = 0
sizeDimension = 0
windowSize = 3
with open(embedding, "r") as input:
    for line in input:
            line = line.rstrip()
            wordCount = int(line.split(" ")[0])
            sizeDimension = int(line.split(" ")[1])
            break
s = (wordCount, sizeDimension)
emb = np.zeros(s, dtype=float)
emb2 = np.zeros(s, dtype=float)
embNoNeg = np.zeros(s, dtype=float)
embNoNeg2 = np.zeros(s, dtype=float)
voc = {}
lineNum = 0
with open(embedding, "r") as input:
    for line in input:
        if lineNum == 0:
            lineNum += 1
            continue
        else:
            line = line.rstrip()
            word = line.split("\t")[0]
            if word in voc:
                print("error")
            voc[word] = lineNum-1
            embLine = line.split("\t")[1]
            ind = 0
            for i in embLine.split(" "):
                emb[lineNum - 1][ind] = float(i)
                ind += 1
        lineNum += 1
lineNum = 0
with open(embedding2, "r") as input:
    for line in input:
        if lineNum == 0:
            lineNum += 1
            continue
        else:
            line = line.rstrip()
            word = line.split("\t")[0]
          #  if word in voc:
          #      print("error")
            if voc[word] != lineNum -1:
                print("mismatch of the two embeddings' position")
            # voc[word] = lineNum-1
            embLine = line.split("\t")[1]
            ind = 0
            for i in embLine.split(" "):
                emb2[lineNum - 1][ind] = float(i)
                ind += 1
        lineNum += 1

lineNum = 0
with open(embeddingNoNeg, "r") as input:
    for line in input:
        if lineNum == 0:
            lineNum += 1
            continue
        else:
            line = line.rstrip()
            word = line.split("\t")[0]
            if voc[word] != lineNum -1:
                print("mismatch of the two embeddings' position")

            embLine = line.split("\t")[1]
            ind = 0
            for i in embLine.split(" "):
                embNoNeg[lineNum - 1][ind] = float(i)
                ind += 1
        lineNum += 1
lineNum = 0
with open(embeddingNoNeg2, "r") as input:
    for line in input:
        if lineNum == 0:
            lineNum += 1
            continue
        else:
            line = line.rstrip()
            word = line.split("\t")[0]
          #  if word in voc:
          #      print("error")
            if voc[word] != lineNum -1:
                print("mismatch of the two embeddings' position")
            # voc[word] = lineNum-1
            embLine = line.split("\t")[1]
            ind = 0
            for i in embLine.split(" "):
                embNoNeg2[lineNum - 1][ind] = float(i)
                ind += 1
        lineNum += 1
import numpy as np
def normalize(x):
     x_normed = (x - x.min(0)) / x.ptp(0)
     return x_normed
# emb = normalize (emb) # c2
# emb2 = normalize(emb2) # t2
# embNoNeg = normalize(embNoNeg) # c1
# embNoNeg2 = normalize(embNoNeg2) # t1


#a = emb @ np.linalg.pinv(embNoNeg) # c1 * (c2)^(-1)
#whetherOrtho = a @ np.transpose(a) # (c1 c2^-1) (c2^-1T c1^T)
#a = a @ emb2  # (c1 * c2^-1) t2
c1 = embNoNeg
t1 = embNoNeg2
c2 = emb
t2 = emb2
#c1TC2
c1Tc2 = np.transpose(c1) @ c2
c2Tc2 = np.transpose(c2) @ c2
c1Tc2_c2Tc2_1 = c1Tc2 @ np.linalg.pinv(c2Tc2)
a = t1 @ c1Tc2_c2Tc2_1 # RT1



sigmoidOurs = c2 @ np.transpose(t2)
sigmoidNoNeg = c1 @ np.transpose(t1)
import math
def sigmoid(xS):
    x = float(xS)
    if x > 6:
        # print("dayu6")
        return 1
    if x < -6:
        # print("xiaoyu-6")
        return 0

    return 1 / (1 + math.exp(-x))
myfunc_vec = np.vectorize(sigmoid)

listofDistBefSig = list()
listofRelDist = list()
listofLogRelDist = list()
diff = list()
e1 = 0
e2 = 0
# for i in range(embNoNeg2.shape[0]):
#     for j in range(embNoNeg2.shape[0]):
#         resultNoNeg = sigmoid(sigmoidNoNeg[i][j])
#         resultOurs = sigmoid(sigmoidOurs[i][j])
#         if resultNoNeg > 0 and resultNoNeg <1 :
#             e1+=1
#         if resultOurs > 0 and resultOurs < 1:
#             e2+=1
#         # if resultNoNeg != 0:
#         diff.append(abs(resultNoNeg - resultOurs))
print(e1)
print(np.mean(diff))
large1 = 0

for i in range(embNoNeg2.shape[0]):
    for j in range(embNoNeg2.shape[1]):
        dist = abs(sigmoidOurs[i][j] - sigmoidNoNeg[i][j])
        listofDistBefSig.append(dist)
        rel = dist/float(abs(sigmoidNoNeg[i][j]))
        listofRelDist.append(rel)
        relLog = math.log(rel)
        listofLogRelDist.append(relLog)

# for i in range(embNoNeg2.shape[0]):
#     dist = np.linalg.norm(sigmoidOurs[i] - sigmoidNoNeg[i], 2)
#     if dist > 0.1:
#         large1+=1
#     listofDistBefSig.append(dist)
print("large 1")
print(large1)

listofDist = list()


for i in range(embNoNeg2.shape[0]):
    dist = np.linalg.norm(a[i] - t2[i], 2)
    listofDist.append(dist)

listofDistNoRotate = list()
for i in range(embNoNeg2.shape[0]):
    dist = np.linalg.norm(t1[i] - t2[i], 2)
    listofDistNoRotate.append(dist)
# for i in range(embNoNeg2.shape[0]):
#     dist = np.linalg.norm(emb[i] - embNoNeg[i])
#     listofDist.append(dist)

listofNewDist = list()

# for i in range(embNoNeg2.shape[0]):
#     dist = np.linalg.norm(embNoNeg2[i])
#     listofNewDist.append(dist)
for i in range(embNoNeg2.shape[0]):
    dist = np.linalg.norm(embNoNeg2[i], 2)
    listofNewDist.append(dist)

listofC2T2 = list()
for i in range(sigmoidOurs.shape[0]):
    dist = np.linalg.norm(sigmoidOurs[i], 2)
    listofC2T2.append(dist)
listofT2 = list()

listofC1T1 = list()
for i in range(sigmoidNoNeg.shape[0]):
    dist = np.linalg.norm(sigmoidNoNeg[i], 2)
    listofC1T1.append(dist)
listofT2 = list()


for i in range(emb2.shape[0]):
    dist = np.linalg.norm(emb2[i], 2)
    listofT2.append(dist)

listofA = list()
for i in range(a.shape[0]):
    dist = np.linalg.norm(a[i], 2)
    listofA.append(dist)


listofC1 = list()
for i in range(a.shape[0]):
    dist = np.linalg.norm(embNoNeg[i], 2)
    listofC1.append(dist)

listofC2 = list()
for i in range(a.shape[0]):
    dist = np.linalg.norm(emb[i], 2)
    listofC2.append(dist)


# print("|C1|: %f" % np.mean(listofC1))
#print("|T1|: %f\n" % np.mean(listofDist))
#
# print("|C2|: %f" % np.mean(listofC2))
#print("|T2|: %f\n" % np.mean(listofT2))
#
print("|R T1| ~ |T2|: %f" % (np.mean(listofA)))
#print("|R T2 - T1|: %f" % np.mean(listofNewDist))
print("|T2 - T1|: %f" % np.mean(listofDistNoRotate))
print("|C2T2 - C1T1|: %f" % np.mean(listofDistBefSig
            ))
print("|R T1 - T2|: %f" % np.mean(listofDist))
print("|T1|: %f\n" % np.mean(listofNewDist))
print("|T2|: %f\n" % np.mean(listofT2))

print("|C1T1|: %f\n" % np.mean(listofC1T1))
print(" C2T2 : %f \n" % np.mean(listofC2T2))
print("|T1|: %f\n" % np.mean(listofNewDist))

import matplotlib.pyplot as plt
import numpy as np

listofDistBefSig = sorted(listofDistBefSig)
plt.hist(listofDistBefSig, density=True, bins=np.arange(start = 0, stop = 10, step = 0.2))

plt.ylabel('Abs Diff Prob')

plt.show()

# print("count of smaller than %d".format() )

listofRelDist = sorted(listofRelDist)
plt.hist(listofRelDist, density=True, bins=np.arange(start = 0, stop = 10, step = 0.2))

plt.ylabel('Relative Diff Prob')



plt.show()
count = len([i for i in listofRelDist if i <= 2])
print("Count")
print(count)
print(len(listofRelDist))


listofLogRelDist = sorted(listofLogRelDist)
plt.hist(listofLogRelDist, density=True, bins=np.arange(start = 0, stop = 10, step = 0.2))

plt.ylabel('Relative Log Diff Prob')



plt.show()
print(max(listofDistBefSig))
print(min(listofDistBefSig))

print(max(listofRelDist))
print(min(listofRelDist))
# import scipy
# import matplotlib.pyplot as plt
# import seaborn as sns
# norm_cdf = scipy.stats.norm.cdf(listofDistBefSig) # calculate the cdf - also discrete
#
# sns.lineplot(x=listofDistBefSig, y=norm_cdf)
# plt.show()

#print("Relative is %f" % (np.mean(listofDist)/np.mean(listofT2)))
#print("Relative no rotate is %f" % (np.mean(listofDistNoRotate)/np.mean(listofT2)))

