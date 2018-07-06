# -*- coding: utf-8 -*-
"""
@author: murali.sai
---
Notes:
Viewed best in spyder

Good Reading:
[1] https://burakhimmetoglu.com/2017/08/22/time-series-classification-with-tensorflow/
[2] https://blog.cardiogr.am/applying-artificial-intelligence-in-medicine-our-early-results-78bfe7605d32
---
#%% PART 1
# TODO:
# 1. Implement some custom objectives:
#    cross-entropy (done), cosine distance, regression error, hinge loss
# 2. Implement different network architectures to the task of time series classification
#    MLP (done), RNN, CNN, etc.
# 3. Train a classifier for time-series classification task
#%% PART 2
# TODO:
# 1. Classify each data point in D_corrupt.
#    Numpy array named corrupt_labels.npz with labels in [0,1,...,8,9]
#%% PART 3
# TODO:
# 1. Predict the next 25 samples for each data point in D_corrupt.
#    Include the results as a shape = (30000,25)
#    Numpy array named corrupt_prediction.npz.
"""
#%% Libraries
# Custom
import datasets.data_reader
from libraries.helpers import train_test_split, progressBar, progressBarSimple
from libraries import data_analysis
# Main
import numpy as np, os, dill
from matplotlib import pyplot as plt
import tensorflow as tf


#%% 1.1 Load clean data and do data analysis
x, y_labels = datasets.data_reader.read_clean_dataset(summary=False)
y = datasets.data_reader.one_hot(y_labels)
x_train, y_train, x_test, y_test = train_test_split(x, y, train_fraction=0.8)
# visit data_analysis.py file for data analysis
#%% 1.2 Time-Series Classification with Deep Learning. Select one among [1.2.1, 1.2.2, 1.2.3, 1.2.4]
#%% 1.2.1 MLP for time-series classification using tensorflow library #<Implemenetation in Keras already provided>
from models.ts_classification.ts_mlp import Configure_MLP, MLP_tf
# Configure Params
configure = Configure_MLP();
configure.custom_loss = 'categorical_crossentropy' # categorical_crossentropy, cosine_distance, regression_error, hinge_loss
configure.create_folders_();
# tf graph and connections
tf.reset_default_graph();
g_mlp  = tf.Graph();
with g_mlp.as_default():
    model = MLP_tf(configure);
# Make data compatible with the architecture defined; function available in model class itself
x_train_, x_test_ = x_train.copy(), x_test.copy();
x_train_, y_train_ = model.make_data_for_batch_training_MLP(x_train_, y_train)
x_test_, y_test_ = model.make_data_for_batch_training_MLP(x_test_, y_test)
#%% 1.2.2 RNN for Time-Series Classification using tensorflow library
from models.ts_classification.ts_rnn import Configure_RNN, RNN_tf
# Configure Params
configure = Configure_RNN();
configure.custom_loss = 'cosine_distance' # categorical_crossentropy, cosine_distance, regression_error, hinge_loss
configure.create_folders_();
# tf graph and connections
tf.reset_default_graph();
g_rnn  = tf.Graph();
with g_rnn.as_default():
    model = RNN_tf(configure);
# Make data compatible with the architecture defined; function available in model class itself
x_train_, x_test_ = np.expand_dims(x_train,axis=-1), np.expand_dims(x_test,axis=-1);
x_train_, y_train_ = model.make_data_for_batch_training_RNN(x_train_, y_train)
x_test_, y_test_ = model.make_data_for_batch_training_RNN(x_test_, y_test)
#%% 1.2.3 CNN for Time-Series Classification using tensorflow library
from models.ts_classification.ts_cnn import Configure_CNN, CNN_tf
# Configure Params
configure = Configure_CNN();
configure.custom_loss = 'categorical_crossentropy' # categorical_crossentropy, cosine_distance, regression_error, hinge_loss
configure.create_folders_();
# tf graph and connections
tf.reset_default_graph();
g_cnn  = tf.Graph();
with g_cnn.as_default():
    model = CNN_tf(configure);
# Make data compatible with the architecture defined; function available in model class itself
x_train_, x_test_ = np.expand_dims(x_train,axis=-1), np.expand_dims(x_test,axis=-1);
x_train_, y_train_ = model.make_data_for_batch_training_CNN(x_train_, y_train)
x_test_, y_test_ = model.make_data_for_batch_training_CNN(x_test_, y_test)
#%% 1.2.4 FC_RNN for Time-Series Classification using tensorflow library
from models.ts_classification.ts_fc_rnn import Configure_FC_RNN, FC_RNN_tf
# Configure Params
configure = Configure_FC_RNN();
configure.custom_loss = 'categorical_crossentropy' # categorical_crossentropy, cosine_distance, regression_error, hinge_loss
configure.create_folders_();
# tf graph and connections
tf.reset_default_graph();
g_fc_rnn  = tf.Graph();
with g_fc_rnn.as_default():
    model = FC_RNN_tf(configure);
# Make data compatible with the architecture defined; function available in model class itself
x_train_, x_test_ = np.expand_dims(x_train,axis=-1), np.expand_dims(x_test,axis=-1);
x_train_, y_train_ = model.make_data_for_batch_training_FC_RNN(x_train_, y_train)
x_test_, y_test_ = model.make_data_for_batch_training_FC_RNN(x_test_, y_test)
#%% 1.3 Train the model selected from [1.2.1, 1.2.2, 1.2.3, 1.2.4] with one of the three loss function
'''
Notes
1. Parameters such as no. of units in dense layers/rnn cells, learning rate, n_epochs, patience, etc.
   can be easily modified by setting them through model.configure.*
   For ease of implementation, all such parameters are pre-set in the respective configure files
'''
# Set Configuration: To set parameter initializations, visit models.ts_classification.global_params.py
model.configure.batch_size = 64;
model.configure.n_epochs = 200;
model.configure.n_timesteps = x.shape[-1];
model.configure.n_features = 1;
model.configure.n_classes = y.shape[-1];
# Save current configuration
with open(model.configure.configure_save_path, 'wb') as opfile:
    dill.dump(configure, opfile)
    opfile.close()
# Training and Inference of the model with current configuration
test_every, start_epoch = 1, 0; # Define when to do testing and also set the starting epoch
with tf.Session(graph = g_cnn) as sess:
    # Initialize log writer and saver for this session and add current graph to tensorboard
    saver = tf.train.Saver() # Saving Variables and Constants
    writer = tf.summary.FileWriter(model.configure.tf_logs);  # Tensorboard
    writer.add_graph(sess.graph);
    # Initialize or load the (best) Model so far; Useful for restarting training
    if start_epoch==0:
        train_loss_, train_loss_min, train_loss_min_epoch = [], np.inf, 0;
        train_acc_, train_acc_max, train_acc_max_epoch = [], -np.inf, 0;
        test_loss_, test_loss_min, test_loss_min_epoch = [], np.inf, 0;
        test_acc_, test_acc_max, test_acc_max_epoch = [], -np.inf, 0;
        tf.global_variables_initializer().run()
    else:
        saved_path = os.path.join(model.configure.model_save_inference, "model.ckpt")
        saver.restore(sess, saved_path); print("Model restored from path: {}".format(saved_path))
        train_loss_, test_loss_ = train_loss_.to_list(), test_loss_.tolist();
        train_acc_, test_acc_ = train_acc_.tolist(), test_acc_.tolist();
    # Training for loss optimization and saving the best models
    # Early Stopping is available with patience
    for epoch in range(start_epoch, model.configure.n_epochs):
        print('EPOCH: {}'.format(epoch));
        # Training
        train_loss, accuracy = 0, 0;
        for i in range(x_train_.shape[0]):
            result = sess.run([model.update_step,model.loss,model.accuracy,model.preds],
                              feed_dict={model.training:True,model.x_:x_train_[i],model.y_:y_train_[i]})
            progressBar(i,x_train_.shape[0],result[1],result[2])
            train_loss+=result[1];
            accuracy+=result[2];
        train_loss/=x_train_.shape[0]; accuracy/=x_train_.shape[0];
        train_loss_.append(train_loss); train_acc_.append(accuracy)
        print('\n Epoch: {}, Training: Avg. Loss: {:.4f} and Avg. Accuarcy: {:.4f}'.format(epoch,train_loss,accuracy));
        # Validation
        if epoch%test_every==0:
            test_loss, accuracy = 0, 0;
            for i in range(x_test_.shape[0]):
                result = sess.run([model.loss,model.accuracy,model.preds],
                                  feed_dict={model.training:False,model.x_:x_test_[i],model.y_:y_test_[i]})
                progressBar(i,x_test_.shape[0],result[0],result[1])
                test_loss+=result[0];
                accuracy+=result[1];
            test_loss/=x_test_.shape[0]; accuracy/=x_test_.shape[0];
            test_loss_.append(test_loss); test_acc_.append(accuracy)
            print('\n Epoch: {}, Testing: Avg. Loss: {:.4f} and Avg. Accuarcy: {:.4f}'.format(epoch,test_loss,accuracy))
        # Model Saving/ Record data/ Tensorboard Writer
        if True:
            # Save variables and tensors to file
            if epoch>1 and train_loss_[-1]<=train_loss_min:
                train_loss_min_epoch, train_loss_min = epoch, train_loss_[-1]
                train_save_path = saver.save(sess, os.path.join(model.configure.model_save_training, "model.ckpt"))
                print("Model (as best train) saved in path: {} after epoch: {}".format(train_save_path,epoch))
            if epoch>1 and test_loss_[-1]<=test_loss_min:
                test_loss_min_epoch, test_loss_min = epoch, test_loss_[-1];
                infer_path = saver.save(sess, os.path.join(model.configure.model_save_inference, "model.ckpt"))
                print("Model (as best val) saved in path: {} after epoch: {}".format(infer_path,epoch))
            # Patience and EarlyStopping
            if (epoch-test_loss_min_epoch)>model.configure.patience:
                print('configured patience epoch {} reached'.format(model.configure.patience))
                print('last best test loss reported after epoch: {}'.format(test_loss_min_epoch))
                break;
            # Flush the data
            print('Tensorboard logs Saved at {}'.format(writer.get_logdir()))
            writer.flush()
# Stack Results
train_loss_, test_loss_ = np.stack(train_loss_), np.stack(test_loss_);
train_acc_, test_acc_ = np.stack(train_acc_), np.stack(test_acc_);
# Plot results
fig, ax = plt.subplots(1,2)
ax[0].plot(range(len(train_loss_)), train_loss_, 'b', label='train_loss_')
ax[0].plot(range(len(test_loss_)), test_loss_, 'r', label='test_loss_')
ax[0].legend(); ax[0].set_xlabel('EPOCHS'); ax[0].set_ylabel('LOSS'); ax[0].set_title('TRAIN vs TEST LOSS'); ax[0].grid();
ax[1].plot(range(len(train_acc_)), train_acc_, 'b', label='train_acc_')
ax[1].plot(range(len(test_acc_)), test_acc_, 'r', label='test_acc_')
ax[1].legend(); ax[1].set_xlabel('EPOCHS'); ax[1].set_ylabel('ACCURACY'); ax[1].set_title('TRAIN vs TEST ACC'); ax[1].grid();
fig.suptitle('model: {}; loss: {}'.format(model.__class__.__name__,model.configure.custom_loss))
# Save current python environment variables for future references
# <...>


#%% ***********************************************************************************************************************************************
#%% ***********************************************************************************************************************************************
#%% ***********************************************************************************************************************************************
#%% ***********************************************************************************************************************************************

#%% 2.0 Load Data
'''
Since we donot have labels for corrupt data, corrupt data from clean data is generated.
On the assumption that both data sets are derived from same data distributions.
'''
# 2.0.1 Load corrupted dataset
x_c, x_c_len = datasets.data_reader.read_corrupted_dataset(summary=True)
# 2.0.2 Load clean data
x, y_labels = datasets.data_reader.read_clean_dataset(summary=True)
y = datasets.data_reader.one_hot(y_labels)
x_train, y_train, x_test, y_test = train_test_split(x, y, train_fraction=0.8)
# 2.0.3 Now Corrupt the Clean data
mean, std = np.mean(x_c_len), np.std(x_c_len);
x_train_my_c, x_test_my_c = x_train.copy(), x_test.copy();
x_train_my_c_len, x_test_my_c_len = np.zeros(len(x_train_my_c)).astype('int32'), np.zeros(len(x_test_my_c)).astype('int32');
for i in range(x_train_my_c.shape[0]):
    sampled_len = np.max([np.min([x_train_my_c.shape[-1],np.int(np.round(np.random.normal(mean, std)))]),156]);
    start_ind = np.random.randint(low=0,high=x_train_my_c.shape[-1]-sampled_len+1,size=1)[0]; end_ind = start_ind+sampled_len;
    x_train_my_c[i,:end_ind-start_ind] = x_train_my_c[i,start_ind:end_ind]; x_train_my_c[i,end_ind-start_ind:]*=0;
    x_train_my_c_len[i] = end_ind-start_ind;
for i in range(x_test_my_c.shape[0]):
    sampled_len = np.max([np.min([x_test_my_c.shape[-1],np.int(np.round(np.random.normal(mean, std)))]),156]);
    start_ind = np.random.randint(low=0,high=x_test_my_c.shape[-1]-sampled_len+1,size=1)[0]; end_ind = start_ind+sampled_len;
    x_test_my_c[i,:end_ind-start_ind] = x_test_my_c[i,start_ind:end_ind]; x_test_my_c[i,end_ind-start_ind:]*=0;
    x_test_my_c_len[i] = end_ind-start_ind;
fig, ax = plt.subplots(1,2);
ax[0].hist(x_train_my_c_len,bins=10); ax[0].set_title('Sample Lengths: Training Data'); ax[0].grid();
ax[1].hist(x_test_my_c_len,bins=10); ax[1].set_title('Sample Lengths: Test Data'); ax[1].grid();
del mean, std, i, sampled_len, start_ind, end_ind, fig, ax
#%% 2.1 Load the classification model and classify samples
c_data = x_train_my_c #x_train_my_c; #x_test_my_c; #x_c;
c_labels = y_train #y_train; #y_test; #np.zeros([c_data.shape[0],10]);
c_len = x_train_my_c_len #x_train_my_c_len; #x_test_my_c_len; #x_c_len;
#%% 2.1.2 Classification using MLP Model
# Load Model
tf.reset_default_graph();
from models.ts_classification.ts_mlp import Configure_MLP, MLP_tf
config_file_path = './logs/ts_classification/mlp_tf_categorical_crossentropy/model_configs'
with open(config_file_path, 'rb') as opfile:
    configure = dill.load(opfile)
    opfile.close()
print(configure.custom_loss);
# tf graph and connections
tf.reset_default_graph();
g_mlp  = tf.Graph();
with g_mlp.as_default():
    model = MLP_tf(configure);
# Make data compatible with the model
  # Data shapes already compatible
# Infer sample by sample by sending as [batch_size,457] dimensional input because this is MLP and we have to send fixed length
with tf.Session(graph=g_mlp) as sess:
    # Restore variables from disk.
    saver = tf.train.Saver() # To restore Variables and Constants
    saved_path = os.path.join(configure.model_save_inference, "model.ckpt")
    saver.restore(sess, saved_path); print("Model restored from path: {}".format(saved_path))
    predictions, accuracy = [], [];
    for i in range(c_data.shape[0]):
        progressBarSimple(i,c_data.shape[0]);
        result = sess.run([model.preds,model.accuracy],feed_dict={model.training:False,model.x_:c_data[i:i+1,:],model.y_:c_labels[i:i+1,:]})
        predictions.append(result[0]);
        accuracy.append(result[1]);
predictions_MLP, accuracy_MLP = np.stack(predictions), np.mean(accuracy)
predictions_MLP = np.reshape(predictions_MLP, [predictions_MLP.shape[0],predictions_MLP.shape[-1]])
if model.configure.custom_loss=='categorical_crossentropy':
    a = np.exp(predictions_MLP - np.max(predictions_MLP,axis=1).reshape([predictions_MLP.shape[0],1]))
    b = a/np.sum(a,axis=1).reshape([predictions_MLP.shape[0],1])
    predictions_MLP = b; del a, b;
#%% 2.1.2 Classification using RNN Model
# Load Model
tf.reset_default_graph();
from models.ts_classification.ts_rnn import Configure_RNN, RNN_tf
config_file_path = './logs/ts_classification/rnn_tf_categorical_crossentropy/model_configs'
with open(config_file_path, 'rb') as opfile:
    configure = dill.load(opfile)
    opfile.close()
print(configure.custom_loss);
# tf graph and connections
tf.reset_default_graph();
g_rnn  = tf.Graph();
with g_rnn.as_default():
    model = RNN_tf(configure);
# Make data compatible with the model
x_c_expanded = np.expand_dims(c_data,axis=-1)
# Infer sample by sample by sending as [1,x_c_len,1] dimensional input because this is RNN and we have variable lengths
with tf.Session(graph=g_rnn) as sess:
    # Restore variables from disk.
    saver = tf.train.Saver() # To restore Variables and Constants
    saved_path = os.path.join(configure.model_save_inference, "model.ckpt")
    saver.restore(sess, saved_path); print("Model restored from path: {}".format(saved_path))
    predictions, accuracy = [], [];
    for i in range(c_data.shape[0]):
        progressBarSimple(i,c_data.shape[0]);
        result = sess.run([model.preds,model.accuracy],feed_dict={model.training:False,model.x_:x_c_expanded[i:i+1,:c_len[i]],model.y_:c_labels[i:i+1,:]})
        predictions.append(result[0]);
        accuracy.append(result[1]);
predictions_RNN, accuracy_RNN = np.stack(predictions), np.mean(accuracy)
predictions_RNN = np.reshape(predictions_RNN, [predictions_RNN.shape[0],predictions_RNN.shape[-1]])
if model.configure.custom_loss=='categorical_crossentropy':
    a = np.exp(predictions_RNN - np.max(predictions_RNN,axis=1).reshape([predictions_RNN.shape[0],1]))
    b = a/np.sum(a,axis=1).reshape([predictions_RNN.shape[0],1])
    predictions_RNN = b; del a, b;
#%% 2.1.3 Classification using CNN Model
# Load Model
from models.ts_classification.ts_cnn import Configure_CNN, CNN_tf
config_file_path = './logs/ts_classification/cnn_tf_categorical_crossentropy/model_configs'
with open(config_file_path, 'rb') as opfile:
    configure = dill.load(opfile)
    opfile.close()
print(configure.custom_loss);
# tf graph and connections
tf.reset_default_graph();
g_cnn = tf.Graph();
with g_cnn.as_default():
    model = CNN_tf(configure);
# Make data compatible with the model
x_c_expanded = np.expand_dims(c_data,axis=-1)
# Infer sample by sample by sending as [1,457,1] dimensional input because the CNN should be fed with full dimension as in training because of dense layers
with tf.Session(graph=g_cnn) as sess:
    # Restore variables from disk.
    saver = tf.train.Saver() # To restore Variables and Constants
    saved_path = os.path.join(configure.model_save_inference, "model.ckpt")
    saver.restore(sess, saved_path); print("Model restored from path: {}".format(saved_path))
    #print(sess.run([model.global_step])[0])
    predictions, accuracy = [], [];
    for i in range(c_data.shape[0]):
        progressBarSimple(i,c_data.shape[0]);
        result = sess.run([model.preds,model.accuracy],feed_dict={model.training:False,model.x_:x_c_expanded[i:i+1,:],model.y_:c_labels[i:i+1,:]})
        predictions.append(result[0]);
        accuracy.append(result[1]);
predictions_CNN, accuracy_CNN = np.stack(predictions), np.mean(accuracy)
predictions_CNN = np.reshape(predictions_CNN, [predictions_CNN.shape[0],predictions_CNN.shape[-1]])
if model.configure.custom_loss=='categorical_crossentropy':
    a = np.exp(predictions_CNN - np.max(predictions_CNN,axis=1).reshape([predictions_CNN.shape[0],1]))
    b = a/np.sum(a,axis=1).reshape([predictions_CNN.shape[0],1])
    predictions_CNN = b; del a, b;
#%% 2.1.4 Classification using FC_RNN Model
# Load Model
tf.reset_default_graph();
from models.ts_classification.ts_fc_rnn import Configure_FC_RNN, FC_RNN_tf
config_file_path = './logs/ts_classification/fc_rnn_tf_categorical_crossentropy/model_configs'
with open(config_file_path, 'rb') as opfile:
    configure = dill.load(opfile)
    opfile.close()
print(configure.custom_loss);
# tf graph and connections
tf.reset_default_graph();
g_fc_rnn  = tf.Graph();
with g_fc_rnn.as_default():
    model = FC_RNN_tf(configure);
# Make data compatible with the model
x_c_expanded = np.expand_dims(c_data,axis=-1)
# Infer sample by sample by sending as [1,x_c_len,1] dimensional input because this is FC_RNN and we have variable lengths
with tf.Session(graph=g_fc_rnn) as sess:
    # Restore variables from disk.
    saver = tf.train.Saver() # To restore Variables and Constants
    saved_path = os.path.join(configure.model_save_inference, "model.ckpt")
    saver.restore(sess, saved_path); print("Model restored from path: {}".format(saved_path))
    predictions, accuracy = [], [];
    for i in range(c_data.shape[0]):
        progressBarSimple(i,c_data.shape[0]);
        result = sess.run([model.preds,model.accuracy],feed_dict={model.training:False,model.x_:x_c_expanded[i:i+1,:c_len[i]],model.y_:c_labels[i:i+1,:]})
        predictions.append(result[0]);
        accuracy.append(result[1]);
predictions_FC_RNN, accuracy_FC_RNN = np.stack(predictions), np.mean(accuracy)
predictions_FC_RNN = np.reshape(predictions_FC_RNN, [predictions_FC_RNN.shape[0],predictions_FC_RNN.shape[-1]])
if model.configure.custom_loss=='categorical_crossentropy':
    a = np.exp(predictions_FC_RNN - np.max(predictions_FC_RNN,axis=1).reshape([predictions_FC_RNN.shape[0],1]))
    b = a/np.sum(a,axis=1).reshape([predictions_FC_RNN.shape[0],1])
    predictions_FC_RNN = b; del a, b;
#%% 2.1.4 Get class labels and save them as npz
# Additionally, ensemble result can be done with weights to MLP, RNN, CNN, FC_RNN results
[w_MLP, w_RNN, w_CNN, w_FC_RNN] = [0,0,0,1]; #[0.05,0.2,0.05,0.7];
final_predictions = w_MLP*predictions_MLP + w_RNN*predictions_RNN + w_CNN*predictions_CNN + w_FC_RNN*predictions_FC_RNN;
classes = np.argmax(final_predictions,axis=1)
np.savez('corrupt_labels.npz', classes);
del c_data, c_labels, c_len, i;

#%% ***********************************************************************************************************************************************
#%% ***********************************************************************************************************************************************
#%% ***********************************************************************************************************************************************
#%% ***********************************************************************************************************************************************
#%% 3.0.1 Load clean data
'''
-A generic model has been developed for all classes to predict the next 25 samples. More details are avaialable in readme.
-A more effective way would be to train individual model per class
-Seq2Seq Learning with scheduled training has been used in prediction
'''
x, y_labels = datasets.data_reader.read_clean_dataset(summary=True)
y = datasets.data_reader.one_hot(y_labels)
x_train, y_train, x_test, y_test = train_test_split(x, y)
#%% 3.0.2 Some Data Analysis on corrupt data
# from libraries import data_analysis # Please check the file for further details
#%% 3.1.1 Training Model for time-series prediction using Seq2Seq RNN Model with Scheduled Training
from models.ts_prediction.ts_seq2seq import Configure_Seq2Seq, Seq2Seq_tf
# Create Model/ Load Model
configure = Configure_Seq2Seq();
configure.custom_loss = 'regression_error' # categorical_crossentropy, cosine_distance, regression_error, hinge_loss
configure.create_folders_();
model = Seq2Seq_tf(configure);
# Save current configuration
with open(model.configure.configure_save_path, 'wb') as opfile:
    dill.dump(configure, opfile)
    opfile.close()
# Training and inference
generate_new_data_every, test_every, start_epoch = 100, 1, 0; # Define when to do testing and also set the starting epoch
if start_epoch==0: # Useful when restarting the training from an earlier stopped training phase
    train_loss_, train_loss_min, train_loss_min_epoch = [], np.inf, 0;
    test_loss_, test_loss_min, test_loss_min_epoch = [], np.inf, 0;
with tf.Session() as sess:
    # Initialize log writer and saver for this session and add current graph to tensorboard
    saver = tf.train.Saver() # Saving Variables and Constants
    writer = tf.summary.FileWriter(model.configure.tf_logs);  # Tensorboard
    writer.add_graph(sess.graph);
    # Initialize or load the (best) Model so far; Useful for restarting training
    if start_epoch!=0:
        saved_path = os.path.join(model.configure.model_save_inference, "model.ckpt")
        saver.restore(sess, saved_path); print("Model restored from path: {}".format(saved_path))
    else:
        tf.global_variables_initializer().run()
    # Initialize global variables and save the best models as you go training
    # Early Stopping is available with patience
    for epoch in range(model.configure.n_epochs):
        print('\n');print('EPOCH: {}'.format(epoch));
        if epoch%generate_new_data_every==0:
            # Make data compatible with the architecture defined; function available in model class itself
            x_train_, x_train_pred_ = model.make_data_for_batch_training_seq2seq(np.expand_dims(x_train,axis=-1));
            x_test_, x_test_pred_ = model.make_data_for_batch_training_seq2seq(np.expand_dims(x_test,axis=-1));
        reuse_predictions_probability = np.min([1,(epoch)/np.min([model.configure.sc_tr_unity_epoch,configure.n_epochs])]) if model.configure.scheduled_training_linear else 0
        # Training
        train_loss = 0;
        for i in range(x_train_.shape[0]):
            result = sess.run([model.update_step,model.loss],
                              feed_dict={model.training:True,
                                         model.x_:x_train_[i],
                                         model.x_pred_:x_train_pred_[i],
                                         model.reuse_predictions_probability:reuse_predictions_probability})
            progressBar(i,x_train_.shape[0],result[1])
            train_loss+=result[1];
        train_loss/=x_train_.shape[0];
        train_loss_.append(train_loss);
        print('\n');print('Epoch: {}, Training: Avg. Loss: {:.4f}'.format(epoch,train_loss));
        # Validation
        if epoch%test_every==0:
            test_loss = 0;
            for i in range(x_test_.shape[0]):
                result = sess.run([model.loss,model.preds],
                                  feed_dict={model.training:False,
                                             model.x_:x_test_[i],
                                             model.x_pred_:x_test_pred_[i],
                                             model.reuse_predictions_probability:1})
                progressBar(i,x_test_.shape[0],result[0])
                test_loss+=result[0];
            test_loss/=x_test_.shape[0];
            test_loss_.append(test_loss);
            print('\n'); print('Epoch: {}, Testing: Avg. Loss: {:.4f}'.format(epoch,test_loss))
        # Model Saving/ Record data/ Tensorboard Writer
        if True:
            # Save variables and tensors to file
            if epoch>1 and train_loss_[-1]<=train_loss_min:
                train_loss_min_epoch, train_loss_min = epoch, train_loss_[-1]
                train_save_path = saver.save(sess, os.path.join(model.configure.model_save_training, "model.ckpt"))
                print("Model (as best train) saved in path: {} after epoch: {}".format(train_save_path,epoch))
            if epoch>1 and test_loss_[-1]<=test_loss_min:
                test_loss_min_epoch, test_loss_min = epoch, test_loss_[-1];
                infer_path = saver.save(sess, os.path.join(model.configure.model_save_inference, "model.ckpt"))
                print("Model (as best val) saved in path: {} after epoch: {}".format(infer_path,epoch))
            # Patience and EarlyStopping
            if (epoch-test_loss_min_epoch)>model.configure.patience:
                break;
            # Flush the data
            print('Tensorboard logs Saved at {}'.format(writer.get_logdir()))
            writer.flush()
# Stack Results
train_loss_, test_loss_ = np.stack(train_loss_), np.stack(test_loss_);
# Plot results
fig, ax = plt.subplots(1,1)
ax.plot(range(len(train_loss_)), train_loss_, 'b', label='train_loss_')
ax.plot(range(len(test_loss_)), test_loss_, 'r', label='test_loss_')
ax.legend(); ax.set_xlabel('EPOCHS'); ax.set_ylabel('LOSS'); ax.set_title('TRAIN vs TEST LOSS'); ax.grid();
fig.suptitle('model: {}; loss: {}'.format('seq2seq_tf',model.configure.custom_loss))
# Save current python environment variables for future references
# <...>
#%% 3.2 Predict next 25 samples for each given sample in x_c
tf.reset_default_graph();
# Load Model
from models.ts_prediction.ts_seq2seq import Configure_Seq2Seq, Seq2Seq_tf
config_file_path = './logs/ts_prediction/seq2seq_tf_regression_error/model_configs'
with open(config_file_path, 'rb') as opfile:
    configure = dill.load(opfile)
    opfile.close()
print('Loss in the configuration: {}'.format(configure.custom_loss));
model = Seq2Seq_tf(configure);
# Load Data
x_c, x_c_len = datasets.data_reader.read_corrupted_dataset(summary=False)
# Make data compatible with the model
x_c_expanded = np.expand_dims(x_c,axis=-1)
past_time_steps = model.configure.timesteps-model.configure.future_time_steps;
x_c_expanded_last = np.zeros([x_c_expanded.shape[0],past_time_steps,1]);
for i in range(x_c_expanded_last.shape[0]):
    x_c_expanded_last[i,:,:] = x_c_expanded[i:i+1,x_c_len[i]-past_time_steps:x_c_len[i],:];
# Split the dataset into batches of size 128
n_rows = x_c_expanded_last.shape[0];
x_c_expanded_last_batches_ = [];
for i in np.arange(0,n_rows-n_rows%model.configure.batch_size,model.configure.batch_size):
    x_c_expanded_last_batches_.append(x_c_expanded_last[i:i+model.configure.batch_size,:,:])
if n_rows%model.configure.batch_size!=0: # Implies left over samples must be added into a last batch
    x_c_expanded_last_batches_.append(x_c_expanded_last[-model.configure.batch_size:,:,:])
x_c_expanded_last_batches_ = np.stack(x_c_expanded_last_batches_);
# Inference step
with tf.Session() as sess:
    # Restore variables from disk.
    saver = tf.train.Saver() # To restore Variables and Constants
    saved_path = os.path.join(model.configure.model_save_inference, "model.ckpt")
    saver.restore(sess, saved_path); print("Model restored from path: {}".format(saved_path))
    predictions = [];
    for i in range(x_c_expanded_last_batches_.shape[0]):
        progressBarSimple(i,x_c_expanded_last_batches_.shape[0]);
        result = sess.run([model.output,model.preds],
                          feed_dict={model.training:False,
                                     model.x_:x_c_expanded_last_batches_[i], #[:128,:100,:], #x_c_expanded[i:i+1,x_c_len[i]-100:x_c_len[i],:],
                                     model.x_pred_:x_c_expanded_last_batches_[i,:,-25:,:], #x_c_expanded_last[:128,:25,:], #x_c_expanded[i:i+1,-25:,:],
                                     model.reuse_predictions_probability:1})
        predictions.append(result[1]); # result[1] will be of size [1,25,1]
predictions = np.vstack(predictions) # [>=30000,25,1]
predictions = np.reshape(predictions, [predictions.shape[0],predictions.shape[1]]); #[>=30000,25]
# Reduce the predictions and x_c_expanded_last_batches_ to size 30000
x_c_expanded_last_batches_ = np.vstack(x_c_expanded_last_batches_);
x_c_expanded_last_batches_ = np.reshape(x_c_expanded_last_batches_, [x_c_expanded_last_batches_.shape[0],x_c_expanded_last_batches_.shape[1]]);
predictions = np.delete(predictions,np.arange(n_rows-(model.configure.batch_size-n_rows%model.configure.batch_size),n_rows),0);
x_c_expanded_last_batches_ = np.delete(x_c_expanded_last_batches_,np.arange(n_rows-(model.configure.batch_size-n_rows%model.configure.batch_size),n_rows),0);
np.savez('corrupt_prediction.npz', predictions);
