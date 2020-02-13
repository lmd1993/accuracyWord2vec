import sys
import numpy as np
UNKNOWN_WORD = "<unk>"
binFN = sys.argv[1] #embedding file
embedding = binFN

embedding2 = sys.argv[2] # second embedding file
# for window size 1. Each two words a, b, to get the #(a, b)/#(a)
embeddingNoNeg = sys.argv[3]
embeddingNoNeg2 = sys.argv[4]
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


a = embNoNeg @ np.linalg.pinv(emb) # c1 * (c2)^(-1)
whetherOrtho = a @ np.transpose(a) # (c1 c2^-1) (c2^-1T c1^T)
a = a @ emb2  # (c1 * c2^-1) t2
emb2Should = (emb @ np.linalg.pinv(embNoNeg) )@ embNoNeg2 # emb* (embNoNeg)^(-1) * embNoNeg2
listofDist = list()


for i in range(embNoNeg2.shape[0]):
    dist = np.linalg.norm(a[i] - embNoNeg2[i], 2)
    listofDist.append(dist)

listofDistNoRotate = list()
for i in range(embNoNeg2.shape[0]):
    dist = np.linalg.norm(embNoNeg[i] - embNoNeg2[i], 2)
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


print("|C1|: %f" % np.mean(listofC1))
print("|T1|: %f\n" % np.mean(listofDist))

print("|C2|: %f" % np.mean(listofC2))
print("|T2|: %f\n" % np.mean(listofT2))

print("|R T2| ~ |T1|: %f" % (np.mean(listofA)))
print("|R T2 - T1|: %f" % np.mean(listofNewDist))
print("|T2 - T1|: %f" % np.mean(listofDistNoRotate))


# print("|R T2 - T1|: %f" % np.mean(listofDist)) #
# print("|T1|: %f\n" % np.mean(listofNewDist)) # no negative sampling; embNoNeg2
# print("|T2|: %f\n" % np.mean(listofT2)) # our embedding; emb2
# print("Relative is %f" % np.mean(listofNewDist)/np.mean(listofDist))

#|RT1- T2|, relative for t2
# for our method, fix one and keep training
# for their method, fix one and keep training