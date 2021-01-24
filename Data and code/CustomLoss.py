# Master Thesis Daan de Jong
# Negation scope detection with Neural Networks

# Loss function

import tensorflow as tf
        
def NegLogLikelihood(y_true, y_pred):
    #remove the padding values (=2)
    yt, yp = y_true[y_true!=2], y_pred[y_true!=2]
    yt, yp = tf.cast(yt, 'float32'), tf.cast(yp, 'float32')

    yp_log1 = tf.math.log(yp) #logits
    yp_log0 = tf.math.log(1-yp) #logits
    prob = tf.math.multiply(yt, yp_log1)+tf.math.multiply(1-yt, yp_log0)
    total = tf.math.reduce_mean(prob)

    return -total
