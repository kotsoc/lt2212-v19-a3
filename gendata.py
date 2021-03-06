import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
import nltk
import random
import pickle
from sklearn.feature_extraction import DictVectorizer

# gendata.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here. You may not use the
# scikit-learn OneHotEncoder, or any related automatic one-hot encoders.

def createNGrams(text):
    """ Method to Create a World list
    and Ngrams for given text."""
    nGrams = {}
    wordList = []
    k = 0
    for line in text:
        ## Parse the tagged text as tuples of Word/POS Tag
        tokens = [nltk.tag.str2tuple(w) for w in line.split()]
        ## List with just the words
        wordList += [tokens[h][0] for h in range(len(tokens))]
        if len(tokens) >= args.ngram:
            nGram = "<start>"+ " "+tokens[0][0]+" "+tokens[1][0]
            nGrams[nGram] = k
            i = 0
            while(i+args.ngram-1 < len(tokens)):
                nGram = tokens[i][0]
                for j in range(i+1, i+args.ngram):
                    nGram += " "+tokens[j][0]
                if nGram not in nGrams:
                    nGrams[nGram] = k+1
                i += 1
                k += 1
    return nGrams,wordList

def createOneHot(vocabularyNGrams, vocabularyWords):
    """ Creating the One Hot represantions of the 
    Ngram ."""
    trainList = []
    k = len(vocabularyWords)
    print(k)
    print(len(vocabularyNGrams))
    for key,value in vocabularyNGrams.items():
        nSplit = key.split()
        oneHot1 = [0]*(k)
        oneHot2 = [0]*(k)
        oneHot1[vocabularyWords[nSplit[0]]] =1
        oneHot2[vocabularyWords[nSplit[1]]] =1
        trainList.append([oneHot1,oneHot2,nSplit[2]])
    return trainList

def wordListToDictionary(wordList):
    """ Converting a word list into a dictionary."""
    ##print(wordList)
    #adding start symbol
    k = 0
    vocabularyWords = {"<start>" : k}
    for i in range(0, len(wordList)):
        if wordList[i] not in vocabularyWords:
            k +=1
            vocabularyWords[wordList[i]] = k
    return vocabularyWords

def writeToFile(nGramList, fileExtension):
    """ Function to write the result list to a file """
    with open(args.outputfile+"."+fileExtension, "wb+") as testFile:
        pickle.dump(nGramList, testFile)
    testFile.close()

#### Main Function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert text to features")
    parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int, default=3, help="The length of ngram to be considered (default 3).")
    parser.add_argument("-S", "--start", metavar="S", dest="startline", type=int,
                        default=0,
                        help="What line of the input data file to start from. Default is 0, the first line.")
    parser.add_argument("-E", "--end", metavar="E", dest="endline",
                        type=int, default=None,
                        help="What line of the input data file to end on. Default is None, whatever the last line is.")
    parser.add_argument("-T", "--train", metavar="T", dest="trainlines",
                        type=int, default=None,
                        help="How many lines to use as train data. Randomly selected")
    parser.add_argument("inputfile", type=str,
                        help="The file name containing the text data.")
    parser.add_argument("outputfile", type=str,
                        help="The name of the output file for the feature table.")
    args = parser.parse_args()

    print("Loading data from file {}.".format(args.inputfile))

    #Open text line by line
    text = open(args.inputfile, encoding="utf-8")
    sentences = text.readlines()
    end = 0
    print("Starting from line {}.".format(args.startline))
    if args.endline:
        print("Ending at line {}.".format(args.endline))
        end = args.endline
    else:
        print("Ending at last line of file.")
        end = len(sentences)-1

    print("Constructing {}-gram model.".format(args.ngram))
    #Making a vector of the vocabulary
    sents = sentences[args.startline: end]
    #print(sents)
    result = createNGrams(sents)
    vocabularyNGrams = result[0]
    listWords = result[1]
    ### Creating Test Ngrams
    trainingLineStart = random.randint(args.startline, end-args.trainlines)
    testSents = sentences[trainingLineStart: (trainingLineStart+args.trainlines)]
    testResult = createNGrams(testSents)
    testvocabNGrams = testResult[0]
    testListWords = testResult[1]

    ### Converting the list into a dictionary for test/train
    vocabularyWords = wordListToDictionary(listWords)
    testvocabularyWords = wordListToDictionary(testListWords)
    ### Creating OneHotfor test/train
    trainList = createOneHot(vocabularyNGrams, vocabularyWords)
    testList = createOneHot(testvocabNGrams, testvocabularyWords)

    print(trainList[1])
    print(testList[1])
    print(trainingLineStart)
    #v = DictVectorizer()
    #matrix = v.fit_transform(nGrams)

    print("Writing table to {}.train".format(args.outputfile))
    writeToFile(trainList, "train")
    print("Writing table to {}.test".format(args.outputfile))
    writeToFile(testList, "test")
    text.close()

# THERE ARE SOME CORNER CASES YOU HAVE TO DEAL WITH GIVEN THE INPUT
# PARAMETERS BY ANALYZING THE POSSIBLE ERROR CONDITIONS.
