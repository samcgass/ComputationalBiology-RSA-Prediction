#   Sam Gass
#   scg0040
#   Computational Biology
#   Project 2
#   Feb 24th 2020
#   This is an ID3 machine learning algorithm for RSA prediction.

import os
import math
import random
import pickle
import copy

#   Node class to create binary tree that will be the model. The
#   true branch is the right branch, and false branch the left.
#   attribute is the data it holds. If it is a leaf node, leaf
#   will be true. If it is a leaf node label will be set to E or
#   B for wheter or not it is exposed or buried.


class Node:
    def __init__(self, attribute=None, label=None, leaf=False):
        self.trueBranch = None
        self.falseBranch = None
        self.attribute = attribute
        self.label = label
        self.leaf = leaf


#   lists for the data. All the data, 75% for training, 25% for testing.
dataSet = []
trainingData = []
testingData = []
#   Dictionary for the table for each amino acid and their attributes. Order of attributes:
#   Hydrophobic / Polar / Positive / Negative / Charged / Small / Tiny / Aromatic / Aliphatic / Proline
table = {
    'A': [True, False, False, False, False, True, True, False, False, False],
    'C': [True, False, False, False, False, True, False, False, False, False],
    'D': [False, True, False, True, True, True, False, False, False, False],
    'E': [False, True, False, True, True, False, False, False, False, False],
    'F': [True, False, False, False, False, False, False, True, False, False],
    'G': [True, False, False, False, False, True, True, False, False, False],
    'H': [False, True, True, False, True, False, False, True, False, False],
    'I': [True, False, False, False, False, False, False, False, True, False],
    'K': [False, True, True, False, True, False, False, False, False, False],
    'L': [True, False, False, False, False, False, False, False, True, False],
    'M': [True, False, False, False, False, False, False, False, False, False],
    'N': [False, True, False, False, False, True, False, False, False, False],
    'P': [True, False, False, False, False, True, False, False, False, True],
    'Q': [False, True, False, False, False, False, False, False, False, False],
    'R': [False, True, True, False, True, False, False, False, False, False],
    'S': [False, True, False, False, False, True, True, False, False, False],
    'T': [True, True, False, False, False, True, False, False, False, False],
    'V': [True, False, False, False, False, True, False, False, True, False],
    'W': [True, False, False, False, False, False, False, True, False, False],
    'Y': [True, True, False, False, False, False, False, True, False, False]
}
#   Paths to directory with fasta and sa files.
fastaPath = 'C:\\Users\\bamas\\Documents\\2020 Spring Semester\\Computational Biology\\Project 2\\fasta\\'
saPath = 'C:\\Users\\bamas\\Documents\\2020 Spring Semester\\Computational Biology\\Project 2\\sa\\'

#   converts all data from fasta and sa files into one list, dataSet, with each data point being a
#   tuple of 11 boolean values, the first ten cooresponding to the attributes from the table for that
#   amino acid and the last value being true if the amino acid is exposed, false otherwise. Once dataSet
#   is complete it will divide the data into training and testing data.


def filesToList():
    seqList = []
    filenames = os.listdir(fastaPath)
    for name in filenames:
        fastaName = fastaPath + name
        saName = saPath + name[:-6] + ".sa"
        with open(fastaName, 'r') as fastaFile, open(saName, 'r') as saFile:
            for fline, sline in zip(fastaFile, saFile):
                for f, s in zip(fline, sline):
                    if f == '>' or s == '>':
                        break
                    if f == '\n' or s == '\n':
                        continue
                    seqList.append((f, s))
    for pair in seqList:
        dataSet.append(aminoToBinary(pair))
    #   divide data
    global trainingData
    global testingData
    trainingData = list(dataSet)
    random.shuffle(trainingData)
    random.shuffle(trainingData)
    length = 0.25 * len(trainingData)
    while length > 0:
        testingData.append(trainingData.pop())
        length -= 1
    return

#   takes a tuple that is char for an amino acid and then a char for whether or not
#   that amino acid is exposed and converts it into a tuple of 11 boolean values.


def aminoToBinary(pair):
    tempList = list(table[pair[0]])
    lastBit = False
    if pair[1] == 'E':
        lastBit = True
    tempList.append(lastBit)
    return tuple(tempList)

#   takes in a data set, the number of positive and negative values in it, p and n, and
#   calculates and returns a list with the information gain of each attribute on the given data


def entropy(examples, p, n):
    hList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    hOfSet = (-p/(p+n)*math.log(p/(p+n), 2))-(n/(p+n)*math.log(n/(p+n), 2))
    for i in range(len(hList)):
        pt = 0
        nt = 0
        pf = 0
        nf = 0
        for example in examples:
            if example[i]:
                if example[-1]:
                    pt += 1
                else:
                    nt += 1
            else:
                if example[-1]:
                    pf += 1
                else:
                    nf += 1
        if pt == 0 and nt == 0:
            hList[i] = 0.0
            continue
        else:
            hOfTrue = (-pt/(pt+nt)*math.log(pt/(pt+nt), 2)) - \
                (nt/(pt+nt)*math.log(nt/(pt+nt), 2))
        if pf == 0 and nf == 0:
            hList[i] = 0.0
            continue
        else:
            hOfFalse = (-pf/(pf+nf)*math.log(pf/(pf+nf), 2)) - \
                (nf/(pf+nf)*math.log(nf/(pf+nf), 2))
        AvgInfo = ((pt+nt)/(p+n)*hOfTrue)+((pf+nf)/(p+n)*hOfFalse)
        gain = hOfSet - AvgInfo
        hList[i] = gain
    return hList

#   the ID3 algorithm. Takes a data set, the current attribute, and list of which
#   attributes have not yet been used and then uses the best attribute to classify
#   the data and recursive calls on itself until the data can no longer be classified.
#   Halts when the binary tree is complete and all leaf nodes have labels and are set to leaves.
#   Returns the root node.


def learn(data, attribute, attributeList):
    root = Node()
    attrList = copy.copy(attributeList)
    p = 0
    n = 0
    for example in data:
        if example[-1]:
            p += 1
        else:
            n += 1
    if n == 0:      # All examples positive
        root.label = 'E'
        root.leaf = True
        return root
    elif p == 0:      # All examples negative
        root.label = 'B'
        root.leaf = True
        return root
    elif all(attr == False for attr in attrList):
        if p > n:
            root.label = 'E'
            root.leaf = True
            return root
        else:
            root.label = 'B'
            root.leaf = True
            return root
    else:
        gainList = entropy(data, p, n)
        # if there is no more info to be gained, return leaf with most common label in data as its label.
        if all(gain == 0 for gain in gainList):
            if p > n:
                root.label = 'E'
                root.leaf = True
                return root
            else:
                root.label = 'B'
                root.leaf = True
                return root
        a = gainList.index(max(gainList))
        if not attrList[a]:
            k = -2
            copyGain = list(gainList)
            copyGain.sort()
            while not attrList[a]:
                a = gainList.index(copyGain[k])
                k -= 1
        attrList[a] = False
        root.attribute = int(a)
        root.trueBranch = Node()
        root.falseBranch = Node()
        trueBranchData = []
        falseBranchData = []
        for example in data:
            if example[a]:
                trueBranchData.append(example)
            else:
                falseBranchData.append(example)
        root.trueBranch = learn(trueBranchData, a, attrList)
        root.falseBranch = learn(falseBranchData, a, attrList)
    return root

#   Takes testing data and the root of a binary tree that represents the
#   model and calculates and prints the percision recall and F1 score.


def testModel(testData, model):
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    for amino in testData:
        m = copy.copy(model)
        while not m.leaf:
            if amino[m.attribute]:
                m = m.trueBranch
            else:
                m = m.falseBranch
        if m.label == 'E':
            if amino[-1]:
                tp += 1
            else:
                fp += 1
        elif m.label == 'B':
            if amino[-1]:
                fn += 1
            else:
                tn += 1
        else:
            print("not a label")
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*((precision*recall)/(precision+recall))
    print("Model Accuracy:")
    print("_______________")
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1: " + str(f1))


#   Trains a model, tests the model, and then pickles the model into RSAmodel.pkl
if __name__ == "__main__":
    filesToList()
    alist = [True, True, True, True, True, True, True, True, True, True]
    a = learn(trainingData, None, alist)
    testModel(testingData, a)
    f = open('RSAmodel.pkl', 'wb')
    p = pickle.Pickler(f)
    s = []
    s.append(a)
    while len(s) != 0:
        current = s.pop()
        p.dump(current)
        if current.trueBranch is not None:
            s.append(current.trueBranch)
        if current.falseBranch is not None:
            s.append(current.falseBranch)
    f.close()
