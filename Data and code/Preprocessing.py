# Master Thesis Daan de Jong
# Negation scope detection with Neural Networks

# Data pre-processing file Using Tensorflow's Preprocessing

# Steps that are carried out in this script
# 1. Create lists containing the cue word(s) and scope word(s) of each sample
# 2. Standardize and tokenize all sentences 
# 3. Create vocabulary and vectorize sentences
# 4. Split data into test and train
# 5. Save as tensorflow dataset


#------------------------------ 0. Set up ------------------------------------#
import numpy as np
import pandas as pd
import re
import random

rawA = pd.read_csv('Original/bioscope_abstract.csv', sep=',')
rawF = pd.read_csv('Original/bioscope_full.csv', sep=',')
raw  = pd.concat([rawA, rawF], ignore_index=True, axis=0)
raw  = raw.drop('sentence_id', axis=1) #sentence_id is considered unimportant
N    = len(raw.index)

#--------------------------- 1. Cues and Scopes ------------------------------#

def now(x): return len(re.findall(' ', x.strip())) + 1 #number of words

#Function to convert a span of char-indices to a indicator vector of the span
def spanVec(sentence, span): 
    n = now(sentence)                        #number of words in the sentence
    spanVec = np.zeros(n)                    #initialize span vector
    
    if type(span)==float:                    #if there is no span to process
        return spanVec
    
    span = re.findall('\d+, \d+', span)      #find the pair(s) of numbers
    span = [eval(x) for x in span]           #list of tuples
    
    for x, y in span:
        #at which word does the span start and end?
        start = now(sentence[0:x]) + 1    
        end   = now(sentence[0:y])
        spanVec[start-1:end] = 1 #mark word indices as 'in the span'
    
    return(spanVec)

cues, scopes = [], []
for i in range(len(raw.index)):
    sen   = raw.loc[i,'sentence']
    cue   = raw.loc[i,'cue_span']
    scope = raw.loc[i,'scope_span']
    cues.append(spanVec(sen, cue))
    scopes.append(spanVec(sen, scope))

#---------------------- 2. Standardization/Tokenization ----------------------#

def standardize(string):
    low = string.lower() #all characters to lower case
    noPunc = re.sub(pattern = r'[^\w\s\-\/]', repl='', string=low)
    return noPunc #return string without punctuation

def tokenize(string): return re.findall("[\w+\-*\w+]+", string) 

sentences    = raw['sentence'].tolist()
standardized = [standardize(x) for x in sentences]
tokenized    = [x.split(sep=' ') for x in standardized]

#Example sentence for report: 169
#For in the report, data descriptives: avg sentence length and avg scope length
avg_sentence_length = sum([len(sentence) for sentence in tokenized])/N
true_scopes = [scope for scope in scopes if sum(scope)>0]
avg_scope_length = sum([sum(scope) for scope in true_scopes])/len(true_scopes)

#--------------------------- 3. Vectorization --------------------------------#

#Get a list of all unique words, with a list of their corresponding frequencies
words, freq = [], []
for s in tokenized:            #loop over the sentences
    for w in s:                #within a sentence, loop over the words
        if not w in words:     #if word not yet in dictionary
            words.append(w)    #add it to the dictionary
            freq.append(1)     #save that this is the first occurence of w
        else:                  #but if word already appeared
            i = words.index(w) #find the position of the word in the dict
            freq[i] += 1       #save that this word occured one more time
            
# Create a Vocabulary of all words, in descending order based on frequency
wordFreqPairs = list(zip(words, freq))
sor = sorted(wordFreqPairs, reverse=True, key=lambda x: x[1]) #sort on freq
#keep only the words and add padding token at the start
V = ['<pad>'] + [x[0] for x in sor]
Vsize = len(V) #17603

#vectorize each sentence by replacing each word by its index in the Vocab
vectorized = [[V.index(w) for w in s] for s in tokenized]

maxSeq = max([len(s) for s in vectorized]) #150

for i in range(N): #post padding with zero's or negative 1's
    #the array of zero's to add at the end of the sequence
    x = np.zeros(maxSeq-len(vectorized[i]), dtype=int)
    #add it to the 
    vectorized[i] = np.append(arr=vectorized[i], values=x)
    cues[i]       = np.append(arr=cues[i],       values=x+2)
    scopes[i]     = np.append(arr=scopes[i],     values=x+2)

#------------------------- 4. Train/Test split -------------------------------#

random.seed(101) #set random seed 
I = list(range(N))
random.shuffle(I)

trainSize = 11500
trainInd  = I[:trainSize]
testInd   = I[trainSize:]
 
senTrain = np.array([vectorized[i] for i in trainInd])
senTest  = np.array([vectorized[i] for i in testInd ])
cueTrain = np.array([cues[i]       for i in trainInd])
cueTest  = np.array([cues[i]       for i in testInd ])
scoTrain = np.array([scopes[i]     for i in trainInd])
scoTest  = np.array([scopes[i]     for i in testInd ])

#---------------------------- 5. Save Datatset -------------------------------#

np.savetxt('senAll.csv', vectorized, fmt='%d', delimiter=',')
np.savetxt('senTrain.csv', senTrain, fmt='%d', delimiter=',')
np.savetxt('senTest.csv' , senTest , fmt='%d', delimiter=',')
np.savetxt('cueTrain.csv', cueTrain, fmt='%d', delimiter=',')
np.savetxt('cueTest.csv' , cueTest , fmt='%d', delimiter=',')
np.savetxt('scoTrain.csv', scoTrain, fmt='%d', delimiter=',')
np.savetxt('scoTest.csv' , scoTest , fmt='%d', delimiter=',')

######################## End of data pre-processing ##########################

