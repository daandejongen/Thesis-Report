# Master Thesis Daan de Jong
# Negation scope detection with Neural Networks

# Custom metrics

import tensorflow as tf
from tensorflow import keras
import numpy as np
import re


class TokenMetrics(keras.metrics.Metric):
    def __init__(self, threshold):
        super(TokenMetrics, self).__init__()
        self.threshold = threshold
        self.tru_pos   = self.add_weight(name='TP', initializer='zeros')
        self.tru_neg   = self.add_weight(name='TN', initializer='zeros')
        self.fal_pos   = self.add_weight(name='FP', initializer='zeros')
        self.fal_neg   = self.add_weight(name='FN', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        true, y_p = y_true[y_true!=2], y_pred[y_true!=2] #remove padding
        pred = tf.where(condition=y_p<self.threshold, x=0.0, y=1.0) #make predictions
        
        tp = tf.where(tf.logical_and(pred==1, true==1), 1., 0.)
        tn = tf.where(tf.logical_and(pred==0, true==0), 1., 0.)
        fp = tf.where(tf.logical_and(pred==1, true==0), 1., 0.)
        fn = tf.where(tf.logical_and(pred==0, true==1), 1., 0.)
        self.tru_pos.assign_add(tf.reduce_sum(tp))
        self.tru_neg.assign_add(tf.reduce_sum(tn))
        self.fal_pos.assign_add(tf.reduce_sum(fp))
        self.fal_neg.assign_add(tf.reduce_sum(fn))

    def result(self):
        ppv = tf.divide(self.tru_pos, self.tru_pos+self.fal_pos)
        npv = tf.divide(self.tru_neg, self.tru_neg+self.fal_neg)
        sensitivity = tf.divide(self.tru_pos, self.tru_pos+self.fal_neg)
        specificity = tf.divide(self.tru_neg, self.fal_pos+self.tru_neg)
        return [ppv, npv, sensitivity, specificity,
                self.tru_pos, self.fal_neg, self.tru_neg, self.fal_pos]
        
    def reset_states(self):
        self.tru_pos.assign(0.0)
        self.tru_neg.assign(0.0)
        self.fal_pos.assign(0.0)
        self.fal_neg.assign(0.0)


def scopeMetrics(y_true, y_pred, threshold):
    exact, too_small, too_large, too_early, too_late = 0, 0, 0, 0, 0
    discon, tn, tp, fp, fn, miss = 0, 0, 0, 0, 0, 0
    n_scopes, n_no_scopes = 0, 0

    for i in range(len(y_true)):
    
        true = y_true[i][y_true[i]!=2].astype(int) #remove padding
        y_p  = y_pred[i][y_true[i]!=2] #remove padding
        pred = np.where(y_p<threshold, 0, 1) #make predictions 

        #inspect scope, if any
        n_true = sum(true)            #number of true scope tokens
        n_pred = sum(pred)            #number of predicted scope tokens
        scope_present = n_true > 0    #is there in fact a scope?
        scope_predicted = n_pred > 0  #is there a scope predicted?    
    
        if not scope_present:
            n_no_scopes += 1
            if not scope_predicted:
                tn += 1
            if scope_predicted:
                fp += 1
    
        if scope_present: 
            n_scopes += 1
            if not scope_predicted: 
                fn += 1
            
            if scope_predicted:
                pred_string = ''.join(re.findall('\d', str(pred)))
                true_string = ''.join(re.findall('\d', str(true)))
                discontinuous = re.search('10+1', pred_string)
                partial_correct = np.dot(true, pred) > 0
                
                if not partial_correct:
                    fp   += 1
                    fn   += 1
                    miss += 1

                if partial_correct:
                    tp += 1
                    if (n_true==np.dot(true, pred) and n_true==n_pred):
                        exact += 1
                    elif discontinuous:
                        discon += 1
                    elif not discontinuous:
                        #start of scope
                        true_start = len(re.findall('(0*)1+', true_string)[0])
                        pred_start = len(re.findall('(0*)1+', pred_string)[0])
                        #end of scope index
                        true_end = len(re.findall('1+', true_string)[0]) + true_start
                        pred_end = len(re.findall('1+', pred_string)[0]) + pred_start
                        
                        true_range = set(range(true_start, true_end+1))
                        pred_range = set(range(pred_start, pred_end+1))
                        
                        if true_range.issubset(pred_range):
                            too_large += 1 
                        elif pred_range.issubset(true_range):
                            too_small += 1
                        elif true_start > pred_start:
                            too_early += 1
                        elif true_start < pred_start:
                            too_late += 1
            
            
    return {'exact_matches':    exact, 
           'too_small':         too_small, 
           'too_large':         too_large, 
           'too_early':         too_early, 
           'too_late':          too_late, 
           'discon':            discon, 
           'true_negatives':    tn, 
           'true_positives':    tp, 
           'false_positives':   fp, 
           'false_negatives':   fn, 
           'misses':            miss
           }
    
            
        
################ DEPRECATED! #####################         
class ScopeMetrics(keras.metrics.Metric):
    def __init__(self, threshold):
        super(ScopeMetrics, self).__init__()
        self.threshold = threshold
        self.exact     = self.add_weight(name='ex', initializer='zeros')
        self.too_small = self.add_weight(name='ts', initializer='zeros')
        self.too_large = self.add_weight(name='tl', initializer='zeros')
        self.too_early = self.add_weight(name='sh', initializer='zeros')
        self.too_late  = self.add_weight(name='sh', initializer='zeros')
        self.discon    = self.add_weight(name='sh', initializer='zeros')        
        self.sco_tn    = self.add_weight(name='tn', initializer='zeros')
        self.sco_tp    = self.add_weight(name='tn', initializer='zeros')
        self.sco_fp    = self.add_weight(name='fp', initializer='zeros')
        self.sco_fn    = self.add_weight(name='tn', initializer='zeros')
        self.miss      = self.add_weight(name='mi', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        true = tf.cast(y_true[y_true!=2], 'float32') #remove padding
        y_p  = y_pred[y_true!=2]                     #remove padding
        pred = tf.where(condition=y_p<self.threshold, x=0., y=1.) #make predictions

        #inspect scope, if any
        n_true = tf.reduce_sum(true) #number of true scope tokens
        n_pred = tf.reduce_sum(pred) #number of predicted scope tokens
        scope_present = n_true > 0   #is there in fact a scope?
        scope_predicted = n_pred > 0   #is there a scope predicted?
        
        if not scope_present:
            if n_pred > 0:
                self.sco_fp.assign_add(1.)
            else:
                self.sco_tn.assign_add(1.)
        
        if scope_present: 
            #any scope predicted?
            if not scope_predicted: 
                self.sco_fn.assign_add(1.)
            
            pred_string = tf.strings.join(tf.strings.as_string(pred, precision=0))
            true_string = tf.strings.join(tf.strings.as_string(true, precision=0))
            discontinous = re.search('10+1', pred_string)
            partial_correct = tf.tensordot(true, pred, axes=1) > 0
            
            if discontinous:
                self.discon.assign_add(1.)
            
            if not partial_correct:
                self.sco_fp.assign_add(1.)
                self.sco_fn.assign_add(1.)
                self.miss.assign_add(1.)
            
            if partial_correct:
                self.sco_tp.assign_add(1.)
                if n_true == n_pred:
                    self.exact.assign_add(1.)
                else:
                    #start of scope
                    true_start = len(re.findall('(0*)1+', true_string)[0])
                    pred_start = len(re.findall('(0*)1+', pred_string)[0])
                    #end of scope index
                    true_end = len(re.findall('1+', true_string)[0]) + true_start
                    pred_end = len(re.findall('1+', pred_string)[0]) + pred_start
                    
                    true_range = set(range(true_start, true_end+1))
                    pred_range = set(range(pred_start, pred_end+1))
                    
                    if true_range.issubset(pred_range):
                        self.too_large.assign_add(1.)
                    elif pred_range.issubset(true_range):
                        self.too_small.assign_add(1.)
                    elif true_start > pred_start:
                        self.too_early.assign_add(1.)
                    elif true_start < pred_start:
                        self.too_late.assign_add(1.)
        
    def result(self):
        res = [
            self.exact,
            self.too_small,
            self.too_large,
            self.too_early,
            self.too_late,
            self.discon,
            self.sco_tp,
            self.sco_tn,
            self.sco_fp,
            self.sco_fn,
            self.miss               
            ]
        
        return res
        
    def reset_states(self):
        self.exact.assign(0.0)
        self.too_small.assign(0.0)
        self.too_large.assign(0.0)
        self.too_early.assign(0.0)
        self.too_late.assign(0.0)
        self.discon.assign(0.0)
        self.sco_tp.assign(0.0)
        self.sco_tn.assign(0.0)
        self.sco_fp.assign(0.0)
        self.sco_fn.assign(0.0)
        self.miss.assign(0.0)
        