import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction import DictVectorizer

# gendata.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here. You may not use the
# scikit-learn OneHotEncoder, or any related automatic one-hot encoders.

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
sents = sentences[args.startline: end+1]
vocabularyNgram = {}
listWords = []
k = 0
for line in sents:
    tokens = [nltk.tag.str2tuple(w) for w in line.split()]
    listWords += [tokens[h][0] for h in range(len(tokens))]
    if len(tokens) >= args.ngram:
        nGram = "<start>"+ " "+tokens[0][0]+" "+tokens[1][0]
        vocabularyNgram[nGram] = k
        i = 0
        while(i+args.ngram-1 < len(tokens)):
            nGram = tokens[i][0]
            for j in range(i+1, i+args.ngram):
                nGram += " "+tokens[j][0]
            if nGram not in vocabularyNgram:
                vocabularyNgram[nGram] = k+1
            i += 1
            k += 1
### Converting the list into a dictionary
k = 0
vocabularyWords = {"<start>" : k}
for i in range(1, len(listWords)):
    if listWords[i] not in vocabularyWords:
        vocabularyWords[listWords[i]] = k
        k +=1
#adding start symbol

##training instances
trainList = []
for key,value in vocabularyNgram.items():
    nSplit = key.split()
    oneHot1 = [0]*(k+1)
    oneHot2 = [0]*(k+1)
    oneHot1[vocabularyWords[nSplit[0]]] =1
    oneHot1[vocabularyWords[nSplit[1]]] =1
    trainList.append([oneHot1,oneHot2,nSplit[2]])
print(trainList[1])
#v = DictVectorizer()
#matrix = v.fit_transform(vocabularyNgram)

print("Writing table to {}.".format(args.outputfile))
text.close()
    
# THERE ARE SOME CORNER CASES YOU HAVE TO DEAL WITH GIVEN THE INPUT
# PARAMETERS BY ANALYZING THE POSSIBLE ERROR CONDITIONS.
