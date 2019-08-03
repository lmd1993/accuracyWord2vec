import sys
import numpy as np
UNKNOWN_WORD = "<unk>"
binFN = sys.argv[1] #embedding file
embedding = binFN
outputName = sys.argv[2] #output file
vocab = sys.argv[3] # vocab file
embedding2 = sys.argv[4] # second embedding file
# for window size 1. Each two words a, b, to get the #(a, b)/#(a)
shuffleFile = sys.argv[5]
contextDict = {}  # context word occurance #(a)
context_Word_Dict = {}  # key: context word; value: Dict {word: #}, #(a with different b)
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
"""ideal stats"""
with open('vocab.txt', 'r') as f:
    for line in f:
        if line.split()[0] in dictWords:
            continue
        else:
            dictWords[line.split()[0]] = line.split()[1]
corpusSize = 0
with open('newText2LimitSize', 'r') as f:
    for line in f:
        # print ("1")
        array = []
        wordList = line.split()
        for i in range(0, len(wordList)):
            tmp = ""
            if wordList[i] in dictWords:
                tmp = wordList[i]
                corpusSize+=1
            else:
                continue
#                tmp = UNKNOWN_WORD
            array.append(tmp)

        # with window size generate pair (i, j)
        for i, u in enumerate(array):
            for j, v in enumerate(
                    array[max(i - windowSize, 0):i + windowSize + 1]):
                # u context, v predict
                if u in contextDict:
                    contextDict[u] = contextDict[u] + 1
                else:
                    contextDict[u] = 1
                if u in context_Word_Dict:
                    if v in context_Word_Dict[u]:
                        context_Word_Dict[u][v] = context_Word_Dict[u][v] + 1
                    else:
                        context_Word_Dict[u][v] = 1
                else:
                    context_Word_Dict[u] = {}
                    context_Word_Dict[u][v] = 1

        #     if tmp in contextDict:
        #         contextDict[tmp] = contextDict[tmp] + 1
        #     else:
        #         contextDict[tmp] = 1
        #
        #
        # # print(len(wordList))
        # # print(len(array))
        # for i in range(1, len(array)):
        #
        #     if array[i - 1] in context_Word_Dict:
        #         # ever met the context word
        #         if array[i] in context_Word_Dict[array[i - 1]]:
        #             # ever met the target word with the context word
        #             context_Word_Dict[array[i - 1]][array[i]] = context_Word_Dict[array[i - 1]][array[i]] + 1
        #         else:
        #             # never met the target word with the context word
        #             context_Word_Dict[array[i - 1]][array[i]] = 1
        #     else:  # never met the context word
        #         context_Word_Dict[array[i - 1]] = {}
        #         context_Word_Dict[array[i - 1]][array[i]] = 1
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

"""compare ideal with real"""

def vectorSig(key, key2, emb, emb2):
    #print(key)
    #print(" key2 %s"%key2)
    return sigmoid(emb[voc[key]].dot(emb2[voc[key2]]))


res = []
for key, value in contextDict.items():
    for key2, value2 in context_Word_Dict[key].items():
        #        print(key)
        #        print(key2)
        ideal = 0
        act = vectorSig(key, key2, emb, emb2)
        #ideal = value2 / float(value)
        if value2/float(value) > 0.5:
            ideal = 1.0
        elif value2/float(value)== 0.5:
            continue
        else:
            ideal = 0.0
        #  print("actual%d and ideal%d"%(act, ideal))
        #        input("Press Enter to continue...")
        # ideal = value2/value
        res.append(abs(act - ideal))
#print(res)
print(np.mean(res))
hs = open(shuffleFile,"a")
hs.write("%f \n" % np.mean(res))
hs.close()

with open(outputName, 'w') as f:
    for item in res:
        f.write("%s\n" % item)

print(corpusSize)
