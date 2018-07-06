# -*- coding: utf-8 -*-
"""
@author: murali.sai
"""
# Global params used to intialize parameters in different architecture configure functions

class global_params(object):
    def __init__(self):
        self.batch_size = 128; # no. of samples per batch for parameter update to happen
        self.n_timesteps = 457; # no. of timesteps
        self.n_features = 1; # no. of features per each timestep
        self.n_classes = 10; # no. of classes; generally used as output in each classification architecture

        self.max_gradient_norm = 5; # To make sure parameters aren't updated by a large magnitude (which might also lead to escaping minima)
        self.learning_rate = 0.001; # Initial learning rate
        self.lr_decay_steps = 100000; # After every self.lr_decay_step no. of batches trained, the self.learning rate decays by self.lr_decay_mag
        self.lr_decay_mag = 0.1; # learning rate decay (multiplication) constant


        self.n_epochs = 200; # no. of epochs during training
        self.val_every_ = 1; # interval after which validation is performed on the validation_dataset
        self.patience = 50; # no. of epochs (with no loss improvement) until before terminating the training process

        self.loss_function = 'categorical_crossentropy' # categorical_crossentropy, cosine_distance, regression_error, hinge_loss

        self.parent_folder = '.';