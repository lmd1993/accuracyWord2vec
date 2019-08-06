from input_data import InputData
import numpy
from model import SkipGramModel
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys

import time
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




class Word2Vec:
    def __init__(self,
                 input_file_name,
                 output_file_name,
                 emb_dimension=100,
                 batch_size=500,
                 window_size=3,
                 iteration=3,
                 initial_lr=0.025,
                 min_count=5,
                 noNeg = 0,
                 loss = "l2",
                 boost = 1):
        """Initilize class parameters.

        Args:
            input_file_name: Name of a text data from file. Each line is a sentence splited with space.
            output_file_name: Name of the final embedding file.
            emb_dimention: Embedding dimention, typically from 50 to 500.
            batch_size: The count of word pairs for one forward.
            window_size: Max skip length between words.
            iteration: Control the multiple training iterations.
            initial_lr: Initial learning rate.
            min_count: The minimal word frequency, words with lower frequency will be filtered.

        Returns:
            None.
        """
        self.data = InputData(input_file_name, min_count)
        self.output_file_name = output_file_name
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.window_size = window_size
        self.iteration = iteration
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)
        self.use_cuda = False
        #self.use_cuda = torch.cuda.is_available()
        #if self.use_cuda:
        #    self.skip_gram_model.cuda()
        self.optimizer = optim.SGD(
            self.skip_gram_model.parameters(), lr=self.initial_lr)
        self.boost = boost
        self.loss = loss
        self.noNeg = noNeg

    def train(self, contextDict, context_Word_Dict):
        """Multiple training.

        Returns:
            None.
        """
        sumTime = 0
        pair_count = self.data.evaluate_pair_count(self.window_size)
        batch_count = self.iteration * pair_count / self.batch_size
        process_bar = tqdm(range(int(batch_count)))
        # self.skip_gram_model.save_embedding(
        #     self.data.id2word, 'begin_embedding.txt', self.use_cuda)
        diffForEpoch = []
        for i in process_bar:
            pos_pairs = self.data.get_batch_pairs(self.batch_size,
                                                  self.window_size)
            if self.noNeg == 0:
                # Change the negative sample count here
                neg_v = self.data.get_neg_v_neg_sampling(pos_pairs, 15)
            else:
                neg_v = self.data.get_all_neg(pos_pairs)
            pos_u = [pair[0] for pair in pos_pairs]
            pos_v = [pair[1] for pair in pos_pairs]

            pos_u = Variable(torch.LongTensor(pos_u))
            pos_v = Variable(torch.LongTensor(pos_v))
            neg_v = Variable(torch.LongTensor(neg_v))
            if self.use_cuda:
                pos_u = pos_u.cuda()
                pos_v = pos_v.cuda()
                neg_v = neg_v.cuda()
            start = time.time()
            self.optimizer.zero_grad()
            loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v, self.boost, self.loss, self.noNeg)
            #print(loss.item())

            loss.backward()
            if i * self.batch_size % 10000 == 0:

                    # embedding = self.u_embeddings.weight.data.numpy()
                firstEmbed = self.skip_gram_model.u_embeddings.weight.data.numpy()
                secondEmbed = self.skip_gram_model.v_embeddings.weight.data.numpy()
                word2id = self.data.word2id
                id2word = self.data.id2word

                def vectorSig(key, key2, emb, emb2):
                        # print(key)
                        # print(" key2 %s"%key2)
                        return sigmoid(emb[word2id[key]].dot(emb2[word2id[key2]]))
                res = []
                for key, value in contextDict.items():
                        for key2, value2 in context_Word_Dict[key].items():
                            #        print(key)
                            #        print(key2)
                            ideal = 0
                            act = vectorSig(key, key2, firstEmbed, secondEmbed)
                            ideal = value2 / float(value)
                            res.append(abs(act - ideal))
                diffToIdeal = np.mean(res)
                # print(" Current diff %f" % np.mean(res))
                diffForEpoch.append(diffToIdeal)
            self.optimizer.step()

            process_bar.set_description("Loss: %0.8f, lr: %0.6f" %
                                        (loss.data,
                                         self.optimizer.param_groups[0]['lr']))
            if i * self.batch_size % 100000 == 0:
                print("Come to update:%f" %(i * self.batch_size / 100000))
                lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            end = time.time()
            sumTime+= end -start
        self.skip_gram_model.save_embedding(
            self.data.id2word, self.output_file_name, self.use_cuda)
        print("Total time %d"%sumTime)
        print("Different for different epoch as:")
        for i in range(len(diffForEpoch)):
            print("%d %f" %(i, diffForEpoch[i]))
        # 10 epochs
        if "10" in self.output_file_name:
            self.output_file_name = self.output_file_name[:-1]
        with open(self.output_file_name[:-1]+"log", 'a') as f:
            f.write("TotalTime %d \n" %sumTime)
if __name__ == '__main__':
    import sys
    import numpy as np

    UNKNOWN_WORD = "<unk>"
    # binFN = sys.argv[1]  # embedding file
    # embedding = binFN
    # outputName = sys.argv[2]  # output file
    # vocab = "" # vocab file
    # embedding2 = sys.argv[4]  # second embedding file
    # for window size 1. Each two words a, b, to get the #(a, b)/#(a)
    # shuffleFile = sys.argv[5]
    contextDict = {}  # context word occurance #(a)
    context_Word_Dict = {}  # key: context word; value: Dict {word: #}, #(a with different b)
    dictWords = {} # key: word, value: num appear
    """word embedding loading and vocabulary"""
    wordCount = 0
    sizeDimension = 0
    windowSize = 3
    """ideal stats"""
    with open('vocab.txt', 'r') as f:
        for line in f:
            if line.split()[0] in dictWords:
                continue
            else:
                dictWords[line.split()[0]] = line.split()[1]
    with open('newText2LimitSize', 'r') as f:
        for line in f:
            # print ("1")
            array = []
            wordList = line.split()
            for i in range(0, len(wordList)):
                tmp = ""
                if wordList[i] in dictWords:
                    tmp = wordList[i]
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


    w2v = Word2Vec(input_file_name=sys.argv[1], output_file_name=sys.argv[2], boost = int(sys.argv[3]), iteration = int(sys.argv[4]), noNeg = int(sys.argv[5]), loss = sys.argv[6])
    #w2v.train()
    #start = time.time()
    w2v.train(contextDict, context_Word_Dict)
    #done = time.time()
    #print("time")
    #print(done-start)
#python word2vec.py newText2LimitSize word_embeddingNeg5 1 3
#python word2vec.py newText2LimitSize word_embeddingNeg5Boost 3 3
