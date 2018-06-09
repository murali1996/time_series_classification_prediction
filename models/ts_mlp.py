# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 19:58:47 2018
@author: murali.sai
----
Notes
MLP for Time-Series Classification using tensorflow library
----------
References
[0] ...
"""
import os, numpy as np
import tensorflow as tf
from libraries import losses

class Configure_MLP(object):
    def __init__(self):
        # 1. Architecture
        self.dense_layer_units, self.activation = [256, 128, 64], tf.nn.relu;
        self.dropout_rates = [0.0, 0.2, 0.1];
        self.custom_loss = 'cosine_distance' # categorical_crossentropy, cosine_distance, regression_error, hinge_loss
        # 2. Training and optimization
        self.batch_size, self.n_timesteps, self.n_classes = 128, 457, 10;
        self.max_gradient_norm, self.learning_rate = 5, 0.001;
        self.n_epochs = 10;
        self.patience = 7; # epochs until before terminating the training process
        # 3. Directories, Sub-directories, Paths
        self.main_dir = './logs';
        self.model_dir = os.path.join(self.main_dir, 'mlp_tf_'+self.custom_loss);
        self.model_save_training = os.path.join(self.model_dir, 'train_best')
        self.model_save_inference = os.path.join(self.model_dir, 'infer_best')
        self.tf_logs = os.path.join(self.model_dir, 'tf_logs');
        self.images = os.path.join(self.model_dir, 'images');
    def create_folders_(self):
        dirs = [self.mmain_dir, self.model_dir, self.model_save_training, self.model_save_inference, self.tf_logs, self.images]
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.mkdir(dir_)
class MLP_tf(object):
    def __init__(self, configure):
        self.configure = configure;
        with tf.variable_scope('inputs'):
            self.training = tf.placeholder(tf.bool) # True: training phase, False: testing/inference phase
            self.x_ = tf.placeholder(tf.float32, shape=[None,self.configure.n_timesteps]) # [batch_size,n_timesteps]
            self.y_ = tf.placeholder(tf.float32, shape=[None,self.configure.n_classes]) # [batch_size,n_classes]
        output_ = self.x_; 
        with tf.variable_scope('multi_layer_mlp'):
            for i, units in enumerate(self.configure.dense_layer_units):
                output_ = tf.layers.dense(inputs=output_, units=units, activation=self.configure.activation, name='dense_{}'.format(i))
                output_ = tf.layers.dropout(output_, rate=self.configure.dropout_rates[i], training=self.training, name='dropout_{}'.format(i))
            self.preds = tf.layers.dense(inputs=output_, units=self.configure.n_classes, activation=self.configure.activation, name='predictions')
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
    def make_data_for_batch_training_MLP(self, x_data, y_data): # Inputs:: x_data:[n_samples,n_timesteps] y_data:[n_samples,n_classes]
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