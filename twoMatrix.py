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

a = embNoNeg.dot(np.transpose(emb))
a = a.dot(emb2)
listofDist = list()
for i in range(embNoNeg2.shape[0]):
    dist = np.linalg.norm(a[i] - embNoNeg2[i])
    listofDist.append(dist)

print(np.mean(listofDist))
