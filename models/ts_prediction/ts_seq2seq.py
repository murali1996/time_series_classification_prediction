# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 19:58:47 2018
@author: murali.sai
----
Notes
Seq2Seq for Time-Series Prediction using tensorflow library
----------
References
[0] ...
"""
import os, numpy as np
import tensorflow as tf
from libraries import losses

class Configure_Seq2Seq(object):
    def __init__(self):
        # 1. Architecture (Seq2Seq: RNN_Layers_Encoder+RNN_Layers_Decoder+Dense Layers)
        self.rnn_units, self.state_activation, self.keep_prob = [128, 128], tf.nn.tanh, 0.8;
        assert(np.any(np.array(self.rnn_units)-self.rnn_units[0])==False) #All values must be equal in current architecture
        #self.dense_layer_units, self.dense_activation, self.dropout_rates = [128,], tf.nn.relu, [0.1,];
        #assert(len(self.dense_layer_units)==len(self.dropout_rates))
        self.custom_loss = 'regression_error' # categorical_crossentropy, cosine_distance, regression_error, hinge_loss
        # 2. Training and optimization
        self.batch_size, self.timesteps, self.future_time_steps, self.n_features, self.n_classes = 128, 125, 25, 1, 10; # n_timesteps=future+past
        assert(self.timesteps > self.future_time_steps)
        self.max_gradient_norm, self.learning_rate = 5, 0.001;
        self.n_epochs, self.patience = 200, 50; # epochs until before terminating the training process
        self.scheduled_training_linear, self.sc_tr_unity_epoch = True, 50;
        assert(self.sc_tr_unity_epoch>=2);
        # 3. Directories, Sub-directories, Paths
        self.main_dir = './logs/ts_prediction';
        self.model_dir = os.path.join(self.main_dir, 'seq2seq_tf_'+self.custom_loss);
        self.model_save_training = os.path.join(self.model_dir, 'train_best')
        self.model_save_inference = os.path.join(self.model_dir, 'infer_best')
        self.tf_logs = os.path.join(self.model_dir, 'tf_logs');
        self.images = os.path.join(self.model_dir, 'images');
        self.configure_save_path = os.path.join(self.model_dir,'model_configs');
    def create_folders_(self):
        dirs = [self.main_dir, self.model_dir, self.model_save_training, self.model_save_inference, self.tf_logs, self.images]
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.mkdir(dir_)
class Seq2Seq_tf(object):
    def __init__(self, configure):
        self.configure = configure;
        with tf.variable_scope('inputs'):
            self.training = tf.placeholder(tf.bool) # True: training phase, False: testing/inference phase
            self.x_ = tf.placeholder(tf.float32, shape=[None,None,self.configure.n_features]) # [batch_size,past_time_steps,num_fetaures]
            self.x_pred_ = tf.placeholder(tf.float32, shape=[None,self.configure.future_time_steps,self.configure.n_features]) # [batch_size,future_time_steps,num_fetaures]
            self.reuse_predictions_probability = tf.placeholder(tf.float32, shape=(), name='reuse_predictions_probability') #in range [0,1]
        with tf.variable_scope('multi_layer_rnn_encoder'):
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
            self.output = tf.concat([outputs_forward[:,-1,:],outputs_backward[:,-1,:]],axis=-1) # [batch_size,2*self.configure.rnn_units[-1]]
        with tf.variable_scope('weights_and_biases'):
            self.weights = {'w_1':self.init_weights([2*self.configure.rnn_units[-1],128],'w_1'),
                            'w_2':self.init_weights([128,self.configure.rnn_units[-1]],'w_2'),
                            'w_3':self.init_weights([self.configure.rnn_units[-1],64],'w_3'),
                            'w_4':self.init_weights([64,self.configure.n_features],'w_4')}
            self.bias = {'b_1':self.init_bias([128],'b_1'),
                         'b_2':self.init_bias([self.configure.rnn_units[-1]],'b_2'),
                         'b_3':self.init_bias([64],'b_3'),
                         'b_4':self.init_bias([self.configure.n_features],'b_4')}
        with tf.variable_scope('enc2dec'):
            latent_vec_ = tf.add(tf.matmul(self.output, self.weights['w_1']), self.bias['b_1'], name='latent_vec_')
            latent_vec = tf.add(tf.matmul(latent_vec_, self.weights['w_2']), self.bias['b_2'], name='latent_vec')
        with tf.variable_scope('multi_layer_rnn_decoder'):
            rnn_cells_decoder = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n, activation=self.configure.state_activation) for n in self.configure.rnn_units]
            rnn_stack_decoder = tf.nn.rnn_cell.MultiRNNCell(rnn_cells_decoder)
            decoder_init_state = tuple([(latent_vec, latent_vec)] * len(self.configure.rnn_units))
            curr_input, curr_state, curr_pred, self.preds = None, None, None, [];
            for j in range(self.configure.future_time_steps):
                if j==0:
                    curr_input = tf.zeros([self.configure.batch_size,self.configure.n_features], dtype='float32')
                else:
                    curr_input = tf.cond(tf.convert_to_tensor(np.random.rand(), 'float32') > self.reuse_predictions_probability,
                                         lambda: tf.convert_to_tensor(self.x_pred_[:,j-1,:], dtype='float32'),
                                         lambda: curr_pred)
                curr_state = decoder_init_state if j==0 else curr_state;
                output, curr_state = rnn_stack_decoder(curr_input, curr_state)
                curr_pred_ = tf.add(tf.matmul(output, self.weights['w_3']), self.bias['b_3'], name='curr_pred_')
                curr_pred_ = tf.layers.dropout(curr_pred_, rate=0.1, training=self.training, name='dropout_curr_pred_')
                curr_pred = tf.add(tf.matmul(curr_pred_, self.weights['w_4']), self.bias['b_4'])
                self.preds = tf.expand_dims(curr_pred, axis=1) if j==0 else tf.concat([self.preds, tf.expand_dims(curr_pred, axis=1)], axis=1);
        with tf.variable_scope('loss_and_optimizer'):
            # 1. Loss function
            self.loss = (tf.reduce_sum(getattr(losses, self.configure.custom_loss)(self.y_,self.preds))/tf.cast(tf.shape(self.x_)[0],tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_,1), tf.argmax(self.preds,1)), tf.float32), name='accuracy')
            # 2. Calculate and clip gradients
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.configure.max_gradient_norm)
            # 3. Set learning Rate: Exponential Decay or a constant value
            self.global_step = tf.Variable(0, trainable=False) # global_step just keeps track of the number of batches seen so far
            #self.lr = tf.train.exponential_decay(self.configure.learning_rate, self.global_step, self.configure.max_global_steps_assumed, 0.1)
            self.lr = self.configure.learning_rate;
            # 4. Update weights and biases i.e trainable parameters
            self.update_step = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
            #self.update_step = tf.train.RMSPropOptimizer(self.lr).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
    def init_weights(self, shape, name):
        return tf.Variable(tf.truncated_normal(shape, stddev = 0.1),name=name)
    def init_bias(self, shape, name):
        return tf.Variable(tf.constant(0.0, shape = shape),name=name)
    def make_data_for_batch_training_seq2seq(self, x_data): # Input SHAPES:: x_data:[n_samples,n_timesteps,n_features] #n_timesteps=457 in given data
        #x_data:[n_samples,n_timesteps,n_features]-->[n_samples,timesteps,n_features]
        x_data_modified = np.zeros([x_data.shape[0],self.configure.timesteps,x_data.shape[-1]]);
        for i in range(x_data.shape[0]):
            rand_index = np.random.randint(0,x_data.shape[1]-self.configure.timesteps,1)[0]
            x_data_modified[i,:,:] = x_data[i,rand_index:rand_index+self.configure.timesteps,:]
        x_data = x_data_modified;
        # Dividing into batches
        n_rows = x_data.shape[0];
        x_, x_pred_ = [], [];
        for i in np.arange(0,n_rows-n_rows%self.configure.batch_size,self.configure.batch_size):
            x_.append(x_data[i:i+self.configure.batch_size,:-self.configure.future_time_steps,:])
            x_pred_.append(x_data[i:i+self.configure.batch_size,-self.configure.future_time_steps:,:])
        if n_rows%self.configure.batch_size!=0: # Implies left over samples must be added into a last batch
            x_.append(x_data[-self.configure.batch_size:,:-self.configure.future_time_steps,:])
            x_pred_.append(x_data[-self.configure.batch_size:,-self.configure.future_time_steps:,:])
        x_, x_pred_ = np.stack(x_), np.stack(x_pred_);
        return x_, x_pred_

#with tf.variable_scope('multi_dense_layers'):
#    for i, units in enumerate(self.configure.dense_layer_units):
#        output_ = tf.layers.dense(inputs=output, units=units, activation=self.configure.dense_activation, name='dense_{}_{}'.format(j,i));
#        output_ = tf.layers.dropout(curr_pred, rate =self.configure.dropout_rates[i], training = self.training, name='dropout_{}_{}'.format(j,i));
#    curr_pred = tf.layers.dense(inputs=output_, units=self.configure.n_features, activation=self.configure.dense_activation, name='predictions_{}'.format(j));


