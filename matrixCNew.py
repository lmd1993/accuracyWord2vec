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
# print("|T1|: %f\n" % np.mean(listofDist)) #Wrong
#
# print("|C2|: %f" % np.mean(listofC2))
# print("|T2|: %f\n" % np.mean(listofT2))
#
print("|R T1| ~ |T2|: %f" % (np.mean(listofA)))
# print("|R T2 - T1|: %f" % np.mean(listofNewDist))
print("|T2 - T1|: %f" % np.mean(listofDistNoRotate))

print("|R T1 - T2|: %f" % np.mean(listofDist))
print("|T1|: %f\n" % np.mean(listofNewDist))
print("|T2|: %f\n" % np.mean(listofT2))
#print("Relative is %f" % (np.mean(listofDist)/np.mean(listofT2)))
#print("Relative no rotate is %f" % (np.mean(listofDistNoRotate)/np.mean(listofT2)))

# No Neg 2: 0.01415732547747142 word_embeddingNoNegl2EpochMultiRunRandomSecond10 word_embeddingNoNegl2EpochMultiRunRandomSecond10_2
# No Neg: 0.014156843306371224 word_embeddingNoNegl2EpochMultiRunRandom10 word_embeddingNoNegl2EpochMultiRunRandom10_2
# Our first: 0.014303889964302173 word_embeddingPairs10 word_embeddingPairs10_2
# Our Second: 0.016427690755805156 word_embeddingPairs_Second10 word_embeddingPairs_Second10_2
#python twoMatrixNew.py word_embeddingPairs10 word_embeddingPairs10_2 word_embeddingPairs_Second10 word_embeddingPairs_Second10_2
# |R T1| ~ |T2|: 2140.879906
# |T2 - T1|: 13.596384
# |R T1 - T2|: 2140.890457
# |T1|: 10.209132
#
# |T2|: 7.291157


#python twoMatrixNew.py word_embeddingNoNegl2EpochMultiRunRandomSecond10 word_embeddingNoNegl2EpochMultiRunRandomSecond10_2 word_embeddingNoNegl2EpochMultiRunRandom10 word_embeddingNoNegl2EpochMultiRunRandom10_2

# |R T1| ~ |T2|: 47.480903
# |T2 - T1|: 3.747547
# |R T1 - T2|: 47.376924
# |T1|: 2.876741
#
# |T2|: 2.586507



# python twoMatrixNew.py word_embeddingPairs10 word_embeddingPairs10_2 word_embeddingNoNegl2EpochMultiRunRandom10 word_embeddingNoNegl2EpochMultiRunRandom10_2
# |R T1| ~ |T2|: 122.087320
# |T2 - T1|: 8.035650
# |R T1 - T2|: 122.886668
# |T1|: 2.586507
#
# |T2|: 7.291157