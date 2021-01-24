# Master Thesis Daan de Jong 
# Negation scope detection with Neural Networks

# Script yielding the information in the Table of the data description
# in the methods section, subsection 'data'

import pandas as pd

abstracts = pd.read_csv('Original/bioscope_abstract.csv', sep=',')
full = pd.read_csv('Original/bioscope_full.csv', sep=',')

def descriptives(X):
    # we know that the negation instances have type=str in the cue_span column
    types = list(map(type, X['cue_span'])) 
    nNeg  = sum([a == str for a in types])
    N = len(X.index)
    propNeg = nNeg/N
    return N, nNeg, propNeg


A = descriptives(abstracts)

F = descriptives(full)