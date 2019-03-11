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

#Making a vector of the vocabulary
sents = sentences[args.startline: end+1]
vocabulary = {}
k = 0
for line in sents:
    tokens = [nltk.tag.str2tuple(w) for w in line.split()]
    if len(tokens) >= args.ngram:
        nGram = "<start>"+ " "+tokens[0][0]+" "+tokens[1][0]
        vocabulary[nGram] = k
        i = 0
        while(i+args.ngram-1 < len(tokens)):
            nGram = tokens[i][0]
            for j in range(i+1, i+args.ngram):
                nGram += " "+tokens[j][0]
            if nGram not in vocabulary:
                vocabulary[nGram] = k+1
            i += 1
            k += 1
        
vect = DictVectorizer()
matrix = vect.fit_transform(vocabulary)
print(vocabulary)
print("Constructing {}-gram model.".format(args.ngram))
print("Writing table to {}.".format(args.outputfile))
text.close()
    
# THERE ARE SOME CORNER CASES YOU HAVE TO DEAL WITH GIVEN THE INPUT
# PARAMETERS BY ANALYZING THE POSSIBLE ERROR CONDITIONS.
