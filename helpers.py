# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 18:41:40 2018
@author: murali.sai
"""
import math
import numpy as np
def train_test_split(X, Y, train_fraction=0.8, shuffle=False):
    N = len(Y)
    if shuffle == True:
        from random import shuffle
        idx = shuffle(np.arange(N))
        X = X[idx]
        Y = Y[idx]
    x_train = X[:math.ceil(N*train_fraction)]
    y_train = Y[:math.ceil(N*train_fraction)]
    x_test = X[math.ceil(N*train_fraction):]
    y_test = Y[math.ceil(N*train_fraction):]
    return x_train, y_train, x_test, y_test

import sys
def progressBar(value, endvalue, loss_val=None, accuracy=None, bar_length=20):
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))
        sys.stdout.write("\rPercent: [{0}] {1}% || Loss: {2} || Acc: {3}".format(arrow + spaces, int(round(percent * 100)), loss_val, accuracy))
        sys.stdout.flush()
        return