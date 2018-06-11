# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 18:38:56 2018
@author: murali.sai
"""
#%% Libraries
# Custom
import datasets.data_reader
from libraries.helpers import train_test_split, progressBar, progressBarSimple
# Main
import numpy as np, os, dill
from matplotlib import pyplot as plt
import tensorflow as tf
#from keras.models import Sequential
#from keras.layers import Dense, Activation, Reshape, Flatten, Input
#from keras.layers import Conv1D, MaxPooling1D
#from keras.layers import BatchNormalization, Dropout
#from keras.layers import LSTM, GRU, Masking
#from keras.optimizers import SGD, RMSprop, Adam

#%% PART 1
# TODO:
# 1. Implement custom objectives:
#    cross-entropy (done), cosine distance,
#    regression error, hinge loss
# 2. Implement network architectures:
#    MLP (done), RNN, CNN
# 3. For each architecture, train a classifier using
#    cross-entropy, cosine distance, and hinge loss.
#    Plot loss/accuracy as a function of epoch on one graph.
#    (9 models total)
#%% 1.0.1 Load clean data
x, y_labels = datasets.data_reader.read_clean_dataset(summary=True)
y = datasets.data_reader.one_hot(y_labels)
x_train, y_train, x_test, y_test = train_test_split(x, y)
#%% Some Data Analysis on clean data
from libraries import data_analysis # Please check the file for further details
#%% 1.1 Write custom function
from libraries import losses # Please check the file for further details
#%% 1.2.1 MLP for time-series classification using tensorflow library #<Implemenetation in Keras already provided>
from models.ts_classification.ts_mlp import Configure_MLP, MLP_tf
# Create Model/ Load Model
configure = Configure_MLP();
configure.custom_loss = 'cosine_distance' # categorical_crossentropy, cosine_distance, regression_error, hinge_loss
configure.create_folders_();
model = MLP_tf(configure);
# Make data compatible with the architecture defined; function available in model class itself
x_train_, x_test_ = x_train.copy(), x_test.copy();
x_train_, y_train_ = model.make_data_for_batch_training_MLP(x_train_, y_train)
x_test_, y_test_ = model.make_data_for_batch_training_MLP(x_test_, y_test)
#%% 1.2.2 RNN for Time-Series Classification using tensorflow library
from models.ts_classification.ts_rnn import Configure_RNN, RNN_tf
# Create Model/ Load Model
configure = Configure_RNN();
configure.custom_loss = 'categorical_crossentropy' # categorical_crossentropy, cosine_distance, regression_error, hinge_loss
configure.create_folders_();
model = RNN_tf(configure);
# Make data compatible with the architecture defined; function available in model class itself
x_train_, x_test_ = np.expand_dims(x_train,axis=-1), np.expand_dims(x_test,axis=-1);
x_train_, y_train_ = model.make_data_for_batch_training_RNN(x_train_, y_train)
x_test_, y_test_ = model.make_data_for_batch_training_RNN(x_test_, y_test)
#%% 1.2.3 CNN for Time-Series Classification using tensorflow library
from models.ts_classification.ts_cnn import Configure_CNN, CNN_tf
# Create Model/ Load Model
configure = Configure_CNN();
configure.custom_loss = 'cosine_distance' # categorical_crossentropy, cosine_distance, regression_error, hinge_loss
configure.create_folders_();
model = CNN_tf(configure);
# Make data compatible with the architecture defined; function available in model class itself
x_train_, x_test_ = np.expand_dims(x_train,axis=-1), np.expand_dims(x_test,axis=-1);
x_train_, y_train_ = model.make_data_for_batch_training_CNN(x_train_, y_train)
x_test_, y_test_ = model.make_data_for_batch_training_CNN(x_test_, y_test)
#%% 1.3 Train the model selected from [1.2.1, 1.2.2, 1.2.3] with one of the three loss function
'''
Notes
1. Parameters such as no. of units in dense layers/rnn cells, learning rate, n_epochs, patience, etc.
   can be easily modified by setting them through model.configure.*
   For ease of implementation, all such parameters are pre-set in the respective configure files
2. It is preferable to use model.configure.dense_activation as tf.nn.tanh instead of tf.nn.relu when using 'hinge_loss'
'''
# Set Configuration
model.configure.batch_size = 128;
model.configure.n_timesteps = x.shape[-1];
model.configure.n_features = 1;
model.configure.n_classes = y.shape[-1];
model.configure.n_epochs
# Save current configuration
with open(model.configure.configure_save_path, 'wb') as opfile:
    dill.dump(configure, opfile)
    opfile.close()
# Training and Inference of the model with current configuration
test_every, start_epoch = 1, 0; # Define when to do testing and also set the starting epoch
if start_epoch==0: # Useful when restarting the training from an earlier stopped training phase
    train_loss_, train_loss_min, train_loss_min_epoch = [], np.inf, 0;
    train_acc_, train_acc_max, train_acc_max_epoch = [], -np.inf, 0;
    test_loss_, test_loss_min, test_loss_min_epoch = [], np.inf, 0;
    test_acc_, test_acc_max, test_acc_max_epoch = [], -np.inf, 0;
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
        print('\n');print('Epoch: {}, Training: Avg. Loss: {:.4f} and Avg. Accuarcy: {:.4f}'.format(epoch,train_loss,accuracy));
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
            print('\n'); print('Epoch: {}, Testing: Avg. Loss: {:.4f} and Avg. Accuarcy: {:.4f}'.format(epoch,test_loss,accuracy))
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
fig.suptitle('model: {}; loss: {}'.format('cnn_tf',model.configure.custom_loss))
# Save current python environment variables for future references
# <...>

#%% PART 2
# TODO:
# 1. Classify each data point in D_corrupt.
#    Feel free to use any of the network architectures/objectives above,
#    and to perform any analysis or pre-processing on the data
#    that may improve the classification accuracy.
#    With your submission, include the results as a shape = (30000,)
#    Numpy array named corrupt_labels.npz with labels in [0,1,...,8,9].
# 2. Document the methods/strategies you used in part 2.1 in your README.
#%% 2.0 Load corrupted dataset
x_c, x_c_len = datasets.data_reader.read_corrupted_dataset(summary=True)
#%% 2.1 Classify each sample in x_c
#%% 2.1.2 Classification using MLP Model
# Load Model
from models.ts_classification.ts_mlp import Configure_MLP, MLP_tf
config_file_path = './logs/ts_classification/mlp_categorical_crossentropy/model_configs'
with open(config_file_path, 'rb') as opfile:
    configure = dill.load(opfile)
    opfile.close()
print(configure.custom_loss);
model = MLP_tf(configure);
# Make data compatible with the model
  # Data shapes already compatible
# Infer sample by sample by sending as [batch_size,457] dimensional input because this is MLP and we have to send fixed length
predictions = [];
for i in range(x_c.shape[0]):
    progressBarSimple(i,x_c.shape[0]);
    result = sess.run([model.preds],feed_dict={model.training:False,model.x_:x_c[i:i+1,:],model.y_:None})
    predictions.append(result[0]);



#%% 2.1.2 Classification using RNN Model
# Load Model
from models.ts_classification.ts_rnn import Configure_RNN, RNN_tf
config_file_path = './logs/ts_classification/rnn_categorical_crossentropy/model_configs'
with open(config_file_path, 'rb') as opfile:
    configure = dill.load(opfile)
    opfile.close()
print(configure.custom_loss);
model = RNN_tf(configure);
# Make data compatible with the model
x_c_expanded = np.expand_dims(x_c,axis=-1)
# Infer sample by sample by sending as [1,x_c_len,1] dimensional input because this is RNN and we have variable lengths
predictions = [];
for i in range(x_c.shape[0]):
    progressBarSimple(i,x_c.shape[0]);
    result = sess.run([model.preds],feed_dict={model.training:False,model.x_:x_c_expanded[i:i+1,x_c_len[i],:],model.y_:None})
    predictions.append(result[0]);


#%% PART 3
# TODO:
# 1. Predict the next 25 samples for each data point in D_corrupt.
#    With your submission, include the results as a shape = (30000,25)
#    Numpy array named corrupt_prediction.npz.
'''
A generic model has been developed for all classes to predict the next 25 samples. A more effective way would be to train individual model per class
Seq2Seq Learning with scheduled training has been used in prediction
'''
# 3.0.1 Data Preperation for model training
#labelled = x_c.copy();
#fig_rows, fig_cols = 7, 2
#fig, ax = plt.subplots(fig_rows, fig_cols, figsize=(16, 5))
#row_inds = np.random.choice(labelled.shape[0],fig_rows*fig_cols,replace=False);
#for fig_row in range(fig_rows):
#    for fig_col in range(fig_cols):
#        ax[fig_row,fig_col].plot(labelled[row_inds[fig_row*fig_cols+fig_col],:]);
#        ax[fig_row,fig_col].set_title('{}'.format(row_inds[fig_row*fig_cols+fig_col]),y=0.93);
#for axx in ax.flat: # Hide x labels and tick labels for top plots and y ticks for right plots.
#    axx.label_outer()
#fig.suptitle('Label: {}'.format(label))

# Set configuration and create RNN model template
configure = Configure_Seq2Seq();
configure.create_folders_();
model = Seq2Seq_tf(configure);

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
            x_train_, x_train_pred_ = model.make_data_for_seq2seq(np.expand_dims(x_train,axis=-1));
            x_test_, x_test_pred_ = model.make_data_for_seq2seq(np.expand_dims(x_test,axis=-1));
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
## Stack Results and plot graphs
#train_loss_, test_loss_ = np.stack(train_loss_), np.stack(test_loss);
#train_acc_, test_acc_ = np.stack(train_acc_), np.stack(test_acc_);
#fig, ax = plt.subplots(1,2)
#ax[0].plot(range(len(train_loss_)), train_loss_,'b',range(len(test_loss_)), test_loss_,'b');
#ax[0].set_xlabel('EPOCHS'); ax[0].set_ylabel('LOSS'); ax[0].set_title('TRAIN vs TEST LOSS'); ax[0].grid();
#ax[1].plot(range(len(train_acc_)), train_acc_,'b',range(len(test_acc_)), test_acc_,'b');
#ax[1].set_xlabel('EPOCHS'); ax[1].set_ylabel('ACCURACY'); ax[1].set_title('TRAIN vs TEST ACC'); ax[1].grid();
## Save current python environment variables for future references
## <...>