# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 19:58:47 2018
@author: murali.sai
----
Notes
CNN for Time-Series Classification using tensorflow library
----------
References
[0] ...
"""
import os, numpy as np
import tensorflow as tf
from libraries import losses

class Configure_CNN(object):
    def __init__(self):
        # 1. Architecture
        self.dropout_rate_dense = 0.2;
        self.custom_loss = 'cosine_distance' # categorical_crossentropy, cosine_distance, regression_error, hinge_loss
        # 2. Training and optimization
        self.batch_size, self.n_timesteps, self.n_features, self.n_classes = 128, 457, 1, 10;
        self.max_gradient_norm, self.learning_rate = 5, 0.001;
        self.n_epochs, self.patience = 10, 7; # Patience: epochs (with no loss improvement) until before terminating the training process
        # 3. Directories, Sub-directories, Paths
        self.main_dir = './logs';
        self.model_dir = os.path.join(self.main_dir, 'rnn_tf_'+self.custom_loss);
        self.model_save_training = os.path.join(self.model_dir, 'train_best')
        self.model_save_inference = os.path.join(self.model_dir, 'infer_best')
        self.tf_logs = os.path.join(self.model_dir, 'tf_logs');
        self.images = os.path.join(self.model_dir, 'images');
    def create_folders_(self):
        dirs = [self.mmain_dir, self.model_dir, self.model_save_training, self.model_save_inference, self.tf_logs, self.images]
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.mkdir(dir_)
class CNN_tf(object):
    def __init__(self, configure):
        self.configure = configure;
        with tf.variable_scope('inputs'):
            self.training = tf.placeholder(tf.bool) # True: training phase, False: testing/inference phase
            self.x_ = tf.placeholder(tf.float32, shape=[None,self.configure.n_timesteps,self.configure.n_features]) # [batch_size,n_timesteps,n_features]
            self.y_ = tf.placeholder(tf.float32, shape=[None,self.configure.n_classes]) # [batch_size,n_classes]
        # Batch Normalization can be performed to boost the training phase learning
        with tf.variable_scope('first_set_of_layers'):
			# [batch_size, 457, 1] --> [batch_size, 457, n_filters]
            out_1_1 = tf.layers.conv1d(inputs=self.x_, filters=16, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='out_1_1')
            out_1_2 = tf.layers.conv1d(inputs=self.x_, filters=16, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu, name='out_1_2')
            out_1_3 = tf.layers.conv1d(inputs=self.x_, filters=16, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu, name='out_1_3')
            out_1 = tf.concat([out_1_1,out_1_2,out_1_3], axis=-1, name='out_1') # [batch_size, 457, 48]
            max_pool_1 = tf.layers.max_pooling1d(inputs=out_1, pool_size=2, strides=2, padding='same', name='max_pool_1') # [batch_size, 229, 48]
        with tf.variable_scope('second_set_of_layers'):
			# [batch_size, 229, 48] --> [batch_size, 229, n_filters]
            out_2_1 = tf.layers.conv1d(inputs=max_pool_1, filters=8, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='out_2_1')
            out_2_2 = tf.layers.conv1d(inputs=max_pool_1, filters=6, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu, name='out_2_2')
            out_2_3 = tf.layers.conv1d(inputs=max_pool_1, filters=4, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu, name='out_2_3')
            out_2 = tf.concat([out_2_1,out_2_2,out_2_3], axis=-1, name='out_2') # [batch_size, 229, 18]
            max_pool_2 = tf.layers.max_pooling1d(inputs=out_2, pool_size=2, strides=2, padding='same', name='max_pool_1') # [batch_size, 115, 18]
        with tf.variable_scope('flatten_and dropout_layers'):
            flat_out = tf.reshape(max_pool_2, (-1,115*18), name='flat_out') # Along each batch # [batch_size, 115*18]
            flat_out_dense_1 = tf.layers.dense(inputs=flat_out, units=256, activation=tf.nn.relu, name='flat_out_dense_1')
            flat_out_dense_1 = tf.layers.dropout(flat_out_dense_1, rate=self.configure.dropout_rate_dense, training=self.training, name='flat_out_dense_1_dropout')
            flat_out_dense_2 = tf.layers.dense(inputs=flat_out_dense_1, units=64, activation=tf.nn.relu, name='flat_out_dense_2')
            flat_out_dense_2 = tf.layers.dropout(flat_out_dense_2, rate=self.configure.dropout_rate_dense, training=self.training, name='flat_out_dense_2_dropout')
            self.preds = tf.layers.dense(inputs=flat_out_dense_2, units=self.configure.n_classes, activation=tf.nn.relu, name='predictions')
        with tf.variable_scope('loss_and_optimizer'):
            # Loss function
            self.loss = (tf.reduce_sum(getattr(losses, self.configure.custom_loss)(self.y_,self.preds))/tf.cast(tf.shape(self.x_)[0],tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_,1), tf.argmax(self.preds,1)), tf.float32), name='accuracy')
            # Calculate and clip gradients
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.configure.max_gradient_norm)
            # Optimization and Update
            self.lr = self.configure.learning_rate;
            self.global_step = tf.Variable(0, trainable=False) # global_step just keeps track of the number of batches seen so far
            #self.lr = tf.train.exponential_decay(self.configure.learning_rate, self.global_step, self.configure.max_global_steps_assumed, 0.1)
            self.update_step = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
    def make_data_for_batch_training_CNN(self, x_data, y_data): # Inputs:: x_data:[n_samples,n_timesteps,n_features] y_data:[n_samples,n_classes]
        '''
        Zero Mean Substraction can be done additionally
        '''
        assert(x_data.shape[0]==y_data.shape[0]);
        n_rows = x_data.shape[0];
        x_data_new, y_data_new = [], [];
        for i in np.arange(0,n_rows-n_rows%self.configure.batch_size,self.configure.batch_size):
            x_data_new.append(x_data[i:i+self.configure.batch_size])
            y_data_new.append(y_data[i:i+self.configure.batch_size])
        if n_rows%self.configure.batch_size!=0: # Implies left over samples must be added into a last batch
            x_data_new.append(x_data[-self.configure.batch_size:])
            y_data_new.append(y_data[-self.configure.batch_size:])
        x_data_new = np.stack(x_data_new);
        y_data_new = np.stack(y_data_new);
        return x_data_new, y_data_new