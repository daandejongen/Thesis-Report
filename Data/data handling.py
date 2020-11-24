# Master Thesis Daan de Jong
# Negation scope detection with Neural Networks

# Data pre-processing file

# Packages
import pandas as pd
import numpy as np
import re

#Function to get word from a span (given as type=char) in a sentence
def getWord(sentence, span):
    spanChar = re.findall("\d+", span) #extract the span numbers as chars
    spanInt = [int(x) for x in spanChar] #turn these into integers
    return(sentence[spanInt[0]:spanInt[1]]) #return the chars corresp. to span

# Loading the data
raw = pd.read_csv('bioscope_full.csv', sep = ',')

#Subset of the data containing only the negated sentences
sub = raw[raw['target'] == 1].drop(['sentence_id', 'target'], axis=1)
sub.reset_index(drop = True, inplace = True)
    
#Create three lists, one with sentences, one with cuewords and one with scopes
sentences = sub.loc[:,'sentence']
cueWords = []
scopeWords = []
for i in range(len(sub.index)):
    cueWords.append(getWord(sub.loc[i,'sentence'], sub.loc[i,'cue_span']))
    scopeWords.append(getWord(sub.loc[i,'sentence'], sub.loc[i,'scope_span']))


############ Tokenization ###############################
#########################################################

#Function: Tokenizer
def tokenize(string): 
    return re.findall("\w+\-*\w+", string) 

sentences = [tokenize(x) for x in sentences]
cueWords = [tokenize(x) for x in cueWords]
scopeWords = [tokenize(x) for x in scopeWords]


############ Vectorization ###############################
##########################################################

#indicator vectors: for every word in the sentence 
cueVec = [] #to be filled with 0 if not a cue word and 1 if a cue word
scopeVec = []
for i in range(len(sub.index)):
    cueVec.append(np.array([int(x in cueWords[i]) for x in sentences[i]]))
    scopeVec.append(np.array([int(x in scopeWords[i]) for x in sentences[i]]))

#if a word is in the cue, it should not be in the scope
scopeVec = [scopeVec[i]-cueVec[i] for i in range(len(scopeVec))]

#Now we make a vocabulary with all the words that are used in the data set
#create a list with all words
flat = [word for sentence in sentences for word in sentence]
#remove the duplicate words
vocab = list(set(flat))
#Get a list of vectorized sentences
senVec = np.array([[vocab.index(word) for word in sen] for sen in sentences])

#make all sentences the same length (namely, the length of the longest sentence)
#by adding -1's to the empty spots
#and do the same for the indicator vectors
maxSen = max([len(sen) for sen in senVec]) #77
for i in range(len(senVec)):
    x = np.repeat(-1, maxSen-len(senVec[i]))
    senVec[i] = np.append(arr=senVec[i], values=x)
    cueVec[i] = np.append(arr=cueVec[i], values=x)
    scopeVec[i] = np.append(arr=scopeVec[i], values=x)
        















