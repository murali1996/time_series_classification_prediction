# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 19:58:47 2018
@author: murali.sai
----
Notes
RNN for Time-Series Classification using tensorflow library
----------
References
[0] ...
"""
import os, numpy as np
import tensorflow as tf
from libraries import losses

class Configure_RNN(object):
    def __init__(self):
        # 1. Architecture
        self.rnn_units, self.state_activation = [128, 128], tf.nn.tanh;
        self.keep_prob_rnn, self.drop_rate_dense = 0.8, 0.2;
        self.custom_loss = 'cosine_distance' # categorical_crossentropy, cosine_distance, regression_error, hinge_loss
        # 2. Training and optimization
        self.batch_size, self.n_features, self.n_classes = 128, 1, 10;
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
class RNN_tf(object):
    def __init__(self, configure):
        self.configure = configure;
        with tf.variable_scope('inputs'):
            self.training = tf.placeholder(tf.bool) # True: training phase, False: testing/inference phase
            self.x_ = tf.placeholder(tf.float32, shape=[None,None,self.configure.n_features]) # [batch_size,n_timesteps,n_features]
            self.y_ = tf.placeholder(tf.float32, shape=[None,self.configure.n_classes]) # [batch_size,n_classes]
        with tf.variable_scope('multi_layer_rnn'):
            with tf.variable_scope('forward'):
                rnn_cells_forward = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n, activation=self.configure.state_activation) for n in self.configure.rnn_units]
                rnn_stack_forward = tf.nn.rnn_cell.MultiRNNCell(rnn_cells_forward)
                #rnn_stack_forward = tf.contrib.rnn.DropoutWrapper(rnn_stack_forward, output_keep_prob=self.configure.keep_prob_rnn)
                outputs_forward, state_forward = tf.nn.dynamic_rnn(rnn_stack_forward, self.x_, dtype = tf.float32)
            with tf.variable_scope('backward'):
                x_backward_ = tf.reverse(self.x_, axis=[1], name='x_backward_')
                rnn_cells_backward = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n, activation=self.configure.state_activation) for n in self.configure.rnn_units]
                rnn_stack_backward = tf.nn.rnn_cell.MultiRNNCell(rnn_cells_backward)
                #rnn_stack_backward = tf.contrib.rnn.DropoutWrapper(rnn_stack_backward, output_keep_prob=self.configure.keep_prob_rnn)
                outputs_backward, state_backward = tf.nn.dynamic_rnn(rnn_stack_backward, x_backward_, dtype = tf.float32)
            self.output =tf.concat([outputs_forward[:,-1,:],outputs_backward[:,-1,:]],axis=-1) # [batch_size,2*self.configure.rnn_units[-1]]
        with tf.variable_scope('dense_layers'):
            self.weights = {'w_1':self.init_weights([2*self.configure.rnn_units[-1],256],'w_1'),
                            'w_2':self.init_weights([256,64],'w_2'),
                            'output_w':self.init_weights([64,self.configure.n_classes],'output_w')}
            self.bias = {'b_1':self.init_bias([256],'b_1'),
                         'b_2':self.init_bias([64],'b_2'),
                         'output_b':self.init_bias([self.configure.n_classes],'output_b')}
            dense_1 = tf.add(tf.matmul(self.output,self.weights['w_1']),self.bias['b_1'],name='dense_1')
            dense_1 = tf.layers.dropout(dense_1, rate=self.configure.drop_rate_dense, training=self.training)
            dense_2 = tf.add(tf.matmul(dense_1,self.weights['w_2']),self.bias['b_2'],name='dense_2')
            self.preds = tf.add(tf.matmul(dense_2,self.weights['output_w']),self.bias['output_b'],name='preds')
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
    def init_weights(self, shape, name):
        return tf.Variable(tf.truncated_normal(shape, stddev = 0.1),name=name)
    def init_bias(self, shape, name):
        return tf.Variable(tf.constant(0.0, shape = shape),name=name)
    def make_data_for_batch_training_RNN(self, x_data, y_data): # Inputs:: x_data:[n_samples,n_timesteps,n_features] y_data:[n_samples,n_classes]
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