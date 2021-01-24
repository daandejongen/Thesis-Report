# Master Thesis Daan de Jong 
# Negation scope detection with Neural Networks

# Word2Vec
# vectorized sentences -> skipgrams -> embeddings for all words

import numpy as np
import tensorflow as tf
import tqdm
from tensorflow import keras

def getSkipGrams(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for vocab_size tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence, 
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples 
    # with positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1, 
          num_sampled=num_ns, 
          unique=True, 
          range_max=vocab_size, 
          seed=seed, 
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      negative_sampling_candidates = tf.expand_dims(
          negative_sampling_candidates, 1)

      context = tf.concat([context_class, negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="float32")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels

#Load data (list of sequences)
text = np.loadtxt('senAll.csv', dtype=int, delimiter=',') 
numNegSam = 15
#get the skipgrams
targets, contexts, labels = getSkipGrams(text, 
                             window_size=5, 
                             num_ns=15, 
                             vocab_size=17603, 
                             seed=7) 

BATCH_SIZE = 1024
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)

class Word2Vec(keras.Model):
  def __init__(self, vs, ed, nns): #ed=embedding dimension
    super(Word2Vec, self).__init__()
    self.target_embedding = keras.layers.Embedding(vs, ed,
                                      input_length=1,
                                      name="w2v_embedding")
    self.context_embedding = keras.layers.Embedding(vs, ed, 
                                       input_length=nns+1)
    self.dots = keras.layers.Dot(axes=(3,2))
    self.flatten = keras.layers.Flatten()

  def call(self, pair):
    target, context = pair
    we = self.target_embedding(target)
    ce = self.context_embedding(context)
    dots = self.dots([ce, we])
    return self.flatten(dots)


def w2vLoss(x_logit, y_true): #loss function
      return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, 
                                                     labels=y_true)

#create the model
word2vec = Word2Vec(vs=17603, ed=50, nns=numNegSam)

#compile model
word2vec.compile(optimizer='adam', loss=w2vLoss, metrics=['accuracy'])

#fit model
word2vec.fit(dataset, epochs=10, batch_size=32)

#store embeddings
weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
        
np.savetxt('embeddings.csv', weights, delimiter=',')


