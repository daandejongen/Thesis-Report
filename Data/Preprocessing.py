# Master Thesis Daan de Jong
# Negation scope detection with Neural Networks

# Data pre-processing file Using Tensorflow's Preprocessing

# Steps that are carried out in this script
# 1a. Create a list containing the cue word(s) of each sample
# 1b. Create a list containing the scope word(s) of each sample
# 2. Standardize all sentences and cue/scope words
# 3. Tokenize all sentences and cue/scope words
# 4. Create labels
# 5. Create vocabularies for vectorization
# 6. Vectorize the sentences and the cue words
# 6. Create a cue embedding matrix

#------------------------------ 0. Set up ------------------------------------#
import tensorflow as tf
import numpy as np
import pandas as pd
import re

raw = pd.read_csv('bioscope_abstract.csv', sep=',')
raw = raw.drop('sentence_id', axis=1) #sentence_id is considered unimportant
    
#--------------------------- 1. Cues and Scopes ------------------------------#

def getWord(sentence, span): #Function to get word from a span
    if type(cue)==float: return ''
    spanChar = re.findall('\d+', span)
    spanInt = [int(x) for x in spanChar]
    return(sentence[spanInt[0]:spanInt[1]])

cues, scopes = [], []
for i in range(len(raw.index)):
    sen   = raw.loc[i,'sentence']
    cue   = raw.loc[i,'cue_span']
    scope = raw.loc[i,'scope_span']
    cues.append(getWord(sen, cue))
    scopes.append(getWord(sen, scope))
    
#--------------------------- 2. Standardization ------------------------------#

def standardize(string):
    low = string.lower() #all characters to lower case
    noPunc = re.sub(pattern = r'[^\w\s\-\/]', repl='', string=low)
    return noPunc #return string without punctuation

#Create a list with all the sentences
sentences = raw['sentence'].tolist()
    
#Standardize the sentences, the cue words and the scope words
sentences = [standardize(x) for x in sentences]
cues      = [standardize(x) for x in cues]
scopes    = [standardize(x) for x in scopes]

#---------------------------- 3. Tokenization --------------------------------#

def tokenize(string): return re.findall("\w+\-*\w+", string) 

sentences = [tokenize(x) for x in sentences]
cues      = [tokenize(x) for x in cues]
scopes    = [tokenize(x) for x in scopes]

#------------------------------ 4. Labels ------------------------------------#

labels = [[word in scopes[i] for word in sentences[i]] for i in list(raw.index)]
labels = [np.array(x).astype(int) for x in labels] #convert to 1's and 0's

#--------------------------- 5. Vocabularies ---------------------------------#

# A list of all words (all words appear once, since it was a set)
allWords = list(set([word for sentence in sentences for word in sentence]))
vocabSize = len(allWords) #13614

# A list of all unique cue words
cueWords = list(set([word for cue in cues for word in cue]))

#--------------------------- 6. Vocabularies ---------------------------------#



















