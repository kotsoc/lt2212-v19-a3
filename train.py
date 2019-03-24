import os, sys
import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

# train.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here.

parser = argparse.ArgumentParser(description="Train a maximum entropy model.")
parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int, default=3, help="The length of ngram to be considered (default 3).")
parser.add_argument("datafile", type=str,
                    help="The file name containing the features.")
parser.add_argument("modelfile", type=str,
                    help="The name of the file to which you write the trained model.")

args = parser.parse_args()

print("Loading data from file {}.".format(args.datafile))
itemList = []
with open(args.outputfile+".train", "rb") as rb:
    itemList = pickle.load(rb)
print(itemList)
rb.close()

print("Training {}-gram model.".format(args.ngram))
print("Writing table to {}.".format(args.modelfile))

# YOU WILL HAVE TO FIGURE OUT SOME WAY TO INTERPRET THE FEATURES YOU CREATED.
# IT COULD INCLUDE CREATING AN EXTRA COMMAND-LINE ARGUMENT OR CLEVER COLUMN
# NAMES OR OTHER TRICKS. UP TO YOU.
