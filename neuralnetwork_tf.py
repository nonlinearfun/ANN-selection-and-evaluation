#********************************************************************************
#**
#**
#**  Copyright (c) 2019 Hilarie Sit, hs764@cornell.edu
#**  Developed by Hilarie Sit, Cornell University
#**  All rights reserved.
#**
#**  Redistribution and use in source and binary forms, with or without
#**  modification, are permitted provided that the following conditions are
#**  met:
#**
#**  - Redistributions of source code must retain the above copyright
#**    notice, this list of conditions and the following disclaimer.
#**
#**  - Neither the name of the copyright holders nor the name of Cornell
#**    University may be used to endorse or promote products derived from
#**    this software without specific prior written permission.
#**
#**  Private, research, and institutional usage is without charge.
#**  Distribution of modified versions of this soure code is admissible
#**  UNDER THE CONDITION THAT THIS SOURCE CODE REMAINS UNDER COPYRIGHT OF
#**  THE ORIGINAL DEVELOPERS, BOTH SOURCE AND OBJECT CODE ARE MADE FREELY
#**  AVAILABLE WITHOUT CHARGE, AND CLEAR NOTICE IS GIVEN OF THE MODIFICATIONS.
#**  Distribution of this code as part of a commercial system is permissible
#**  ONLY BY DIRECT ARRANGEMENT WITH THE DEVELOPERS.
#**
#**
#**  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#**  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#**  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#**  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#**  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#**  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#**  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#**  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#**  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#**  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#**  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#**
#********************************************************************************

import numpy as np
import tensorflow as tf
import pandas
import time
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def shuffle(x,y):
    """ Returns shuffled x and y arrays
        [x],[y]         list or 1D array
    """
    x, y = np.array(x), np.array(y)
    n_y = len(y)
    index_array = np.arange(n_y)
    np.random.shuffle(index_array)
    sx, sy = [], []
    for idx, val in enumerate(index_array):
        sx.append(x[val])
        sy.append(y[val])
    sx, sy = np.array(sx), np.array(sy)
    return sx, sy

def batching(x,y,n):
    """ Returns list of batched sub-arrays in a list
        [x],[y]         1D array to be batched
        [n]             number of samples per batch
    """
    n_batches = np.int64(np.floor(len(y)/n))
    x_batch = np.array_split(x,n_batches)
    y_batch = np.array_split(y,n_batches)
    return n_batches, x_batch, y_batch

def return_batch(x_batch,y_batch,counter):
    """ Returns specific batch from list
        [x_batch], [y_batch]            list of batches
        [counter]                       location of batch
    """
    x = x_batch[counter]
    y = y_batch[counter]
    return x, y

class dataset:
    def __init__(self, args):
        self.inputs, self.labels = None, None
        self.x_train_list, self.y_train_list, self.x_test_list, self.y_test_list = [],[],[],[]
        self.no_noise_input, self.no_noise_label = [], []
        self.noise = args.noise
        self.process_data(args)

    def process_data(self, args):
        """ Import dataset from csv and format into feature and labels np float arrays
            [args]          arguments from parser
        """
        df = pandas.read_csv(args.csv+'.csv')
        data = df.values
        m,n = data.shape
        inputs, labels = [], []
        header = list(df)
        for ind, val in enumerate(header):
            inp = np.float32(data[:,ind])
            label = np.float32(val[1:].replace('_','.'))
            # if args.noise is True, add noise; else, append original
            if args.noise:
                noisy = self.add_noise(inp)
                inputs.append(noisy)
                labels.append(label)
                # save copy of arrays with no noise for augmenting process
                self.no_noise_input.append(inp)
                self.no_noise_label.append(label)
            else:
                inputs.append(inp)
                labels.append(label)
        self.inputs, self.labels = shuffle(inputs, labels)
        self.cross_val()

    def add_noise(self, inp):
        """ Create and then add gaussian noise - zero mean and std of 0.1*||inp||_infty
            [inp]           array that needs noise
        """
        inf_norm = max(abs(inp))
        std = 0.1*inf_norm
        noise = np.random.normal(loc=0,scale=std,size=inp.shape)
        return inp+noise

    def cross_val(self):
        """Create list with CV splits"""
        kf = KFold(n_splits=5, shuffle=False)           # already shuffled before calling
        for train_ind, test_ind in kf.split(self.inputs):
            self.x_train_list.append(self.inputs[train_ind])
            self.y_train_list.append(self.labels[train_ind])
            self.x_test_list.append(self.inputs[test_ind])
            self.y_test_list.append(self.labels[test_ind])

    def return_cross_val(self, counter):
        """ Returns specific CV split from list
            [counter]           location of split
        """
        x_train, y_train = self.x_train_list[counter], self.y_train_list[counter]
        x_test, y_test = self.x_test_list[counter], self.y_test_list[counter]
        return x_train, y_train, x_test, y_test

    def training_augment(self, y_train, aug_num):
        """ Returns augmented list
            [aug_num]           number of samples per original datapoint
        """
        x_list, y_list = [], []
        for i in range(len(y_train)):
            label = y_train[i]
            ind = self.no_noise_label.index(y_train[i])
            inp = self.no_noise_input[ind]
            for j in range(aug_num):
                noisy = self.add_noise(inp)
                x_list.append(noisy)
                y_list.append(label)
        ax_train, ay_train = shuffle(x_list, y_list)
        return ax_train, ay_train

def fclayer(x,scope,n):
    """ Returns output from fully connected layer
        [scope]         variable scope
        [n]             number of neurons in next layer
    """
    l = x.get_shape().as_list()[-1]
    with tf.variable_scope(scope):
        w = tf.get_variable('w',initializer=tf.truncated_normal([l,n],stddev=0.01))
        b = tf.get_variable('b',initializer=tf.constant(0,tf.float32,[n]))
    return tf.add(tf.matmul(x,w),b)

class neural_network:
    def __init__(self, x, y, n1, lr, bayes_scope):
        # network architecture
        h1 = tf.nn.relu(fclayer(x,'h1'+bayes_scope,n1))
        self.pred = fclayer(h1,'regression'+bayes_scope,1)
        self.loss = tf.reduce_mean(tf.square(tf.subtract(y, self.pred)))
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
        # calculate accuracy
        difference = tf.abs(self.pred - y)
        correct = tf.round(difference)
        correct_prediction = tf.equal(correct, tf.zeros(shape=tf.shape(correct)))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def run(sess, x_train, y_train, x_test, y_test, n1, lr, bs, n_epochs, bayes_scope, counter, model_num):
    """ Trains model for one CV fold and returns lowest loss and accuracy for the fold
        [sess]              Tensorflow session
        [x_train]           Input array
        [y_train]           Labels
        [x_test]            Input array
        [y_test]            Labels
        [n1],[lr],[bs]      Hyperparameters
        [n_epochs]          Number of epochs
        [bayes_scope]       To help create new graph
        [counter]           Iterate through K fold splits
        [model_num]         Model number
    """
    low_loss = 1000

    # define placeholders for input variables
    _, input_dim = x_train.shape
    x = tf.placeholder(tf.float32,[None,input_dim])
    y = tf.placeholder(tf.float32)

    # call model and initialize graph
    model = neural_network(x,y,n1,lr,bayes_scope)
    sess.run(tf.global_variables_initializer())

    # training
    for i in range(n_epochs):
        n_batches, x_batch, y_batch = batching(x_train, y_train, bs)
        x_batch, y_batch = shuffle(x_batch, y_batch)
        pointer = 0

        for j in range(n_batches):
            xll_train, yll_train = return_batch(x_batch, y_batch, pointer)
            yll_train = np.reshape(yll_train, (-1,1))
            _, train_loss = sess.run([model.train_op, model.loss], feed_dict={x: xll_train, y: yll_train})
            pointer = pointer + 1

        # calculate test loss
        if i % 5 == 0:
            y_test = np.reshape(y_test, (-1,1))
            test_loss, accuracy = sess.run([model.loss, model.accuracy], feed_dict={x: x_test, y: y_test})
            print('epoch'+str(i))
            print('train_loss:'+str(train_loss))
            print('test_loss:'+str(test_loss))
            print('test_acc:'+str(accuracy))
            if low_loss > test_loss:
                low_loss = test_loss
                low_accuracy = accuracy

    return low_loss, low_accuracy
