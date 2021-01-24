# Master Thesis Daan de Jong
# Negation scope detection with Neural Networks

# Defining the model in Keras Tensorflow

#------------------------------ 0. Set up ------------------------------------#
import tensorflow     as tf
import numpy          as np
import random         as rd
from tensorflow       import keras
from tensorflow.keras import layers
from CustomEmbedding  import CueEmbeddingLayer
from CustomLoss       import NegLogLikelihood
from CustomMetric     import TokenMetrics, scopeMetrics

vSize     = 17603 #obtained from Preprocessing file
seqLen    = 150   #chosen in Preprocessing file
threshold = 0.5   #chosen here
SEED = 100          #seed used both globally and in operations (weights inits)

tf.random.set_seed(SEED)

senTrain = np.loadtxt('senTrain.csv', dtype=int, delimiter=',')
cueTrain = np.loadtxt('cueTrain.csv', dtype=int, delimiter=',')
scoTrain = np.loadtxt('scoTrain.csv', dtype=int, delimiter=',')

E = np.loadtxt('embeddings.csv', delimiter=',')
embDim = E.shape[1]    #chosen in word2vec

#----------------------- 1. Defining the model--------------------------------#

input1  = keras.Input(shape=(seqLen), name='sentencesInput')
wordEmb = layers.Embedding(input_dim=vSize, output_dim=embDim,
                          embeddings_initializer=keras.initializers.Constant(E),
                          mask_zero=True, trainable=False)(input1)
x = layers.Bidirectional(layers.LSTM(units=32, activation='tanh',
                         recurrent_activation='sigmoid',
                         kernel_initializer=keras.initializers.GlorotUniform(SEED),
                         return_sequences=True))(wordEmb)

input2 = keras.Input(shape=(seqLen), name='cueInput')
cueEmb = CueEmbeddingLayer(embDim=embDim, mask_value=2)(input2)
y = layers.Bidirectional(layers.LSTM(units=32, activation='tanh',
                         recurrent_activation='sigmoid',
                         kernel_initializer=keras.initializers.GlorotUniform(SEED),
                         return_sequences=True))(cueEmb)

merged  = layers.concatenate(inputs=[x, y])
dense   = layers.Dense(units=64, 
                       activation='relu',
                       kernel_initializer=keras.initializers.GlorotUniform(SEED))(merged)
final   = layers.Dense(units=1, 
                       activation='sigmoid',
                       kernel_initializer=keras.initializers.GlorotUniform(SEED))(dense)
outputs = tf.squeeze(final)
 
model = keras.Model(inputs=[input1, input2], outputs=outputs, name="NegScoDet")

model.summary()


#----------------------- 2. Compiling the model-------------------------------#

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=NegLogLikelihood,
    metrics=[TokenMetrics(threshold=threshold)]
    )

#--------------------------- 3. Train model ----------------------------------#

history = model.fit(
    x={'sentencesInput':senTrain, 'cueInput':cueTrain},
    y=scoTrain, 
    batch_size=32,
    epochs=3, 
    validation_split=.2
    )

#--------------------------- 4. Test model -----------------------------------#

senTest = np.loadtxt('senTest.csv', delimiter=',')
cueTest = np.loadtxt('cueTest.csv', delimiter=',')
scoTest = np.loadtxt('scoTest.csv', delimiter=',')

results = model.evaluate(x={'sentencesInput':senTest, 'cueInput':cueTest},
                         y=scoTest)
predictions = model.predict(x={'sentencesInput':senTest, 'cueInput':cueTest})

#------------------------- 5. Evaluate model ---------------------------------#

#Token Level
precision = results[1][0]
negative_predictive_value = results[1][1]
recall = results[1][2]
specificity = results[1][3]
f1 = (2*precision*recall)/(precision+recall)

#Scope level
scopemetrics = scopeMetrics(y_true=scoTest, 
                            y_pred=predictions, 
                            threshold=threshold)

############################# End of code ####################################