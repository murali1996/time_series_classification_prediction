# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 18:40:42 2018
@author: murali.sai
"""
import tensorflow as tf

def categorical_crossentropy(y_true, y_pred):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    return loss
# Using only basic tensorflow ops, implement the following functions
# TODO: cosine distance
def cosine_distance(y_true, y_pred):
    """
    Cosine_distance_loss = 1 - cosine_similarity
    cosine distance using only basic tensorflow ops.
    (Do not use tf.losses, tf.nn, etc.)
    """
    # Correctness check
    tf.assert_equal(tf.shape(y_true),tf.shape(y_pred), name='assert_cond')
    epsilon = 0.0001
    # Denominator; Norms
    y_true_norm = tf.sqrt(tf.reduce_mean(tf.multiply(y_true,y_true)),name='y_true_norm')
    y_pred_norm = tf.sqrt(tf.reduce_mean(tf.multiply(y_pred,y_pred)),name='y_pred_norm')
    den_ = tf.multiply(y_true_norm,y_pred_norm)+epsilon;
    # Numerator; Cross-multiplication
    cross_multiply = tf.sqrt(tf.reduce_mean(tf.multiply(y_true,y_pred)),name='cross_multiply')
    num_ = cross_multiply+epsilon;
    loss = 1-tf.div(num_,den_)
    return loss
# TODO: regression error
# mean absolute error ^ n
def regression_error(y_true, y_pred, p=2):
    """
    Regression error using only basic tensorflow ops.
    (Do not use tf.losses, tf.nn, etc.)
    """
    abs_diff = tf.abs(tf.subtract(y_true,y_pred))
    loss = tf.reduce_mean(tf.pow(abs_diff,p))
    return loss
# TODO: hinge loss
def hinge_loss(y_true, y_pred):
    """
    Hinge loss loss using only basic tensorflow ops.
    (Do not use tf.losses, tf.nn, etc.)
    """
    return
    #return loss