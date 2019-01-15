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

    def add_noise(self, inp):
        """ Create and then add gaussian noise - zero mean and std of 0.1*||inp||_infty
            [inp]           array that needs noise
        """
        inf_norm = max(abs(inp))
        std = 0.1*inf_norm
        noise = np.random.normal(loc=0,scale=std,size=inp.shape)
        return inp+noise

    def training_augment(self, y_train, args):
        """ Returns augmented list
            [args]           Arguments from parser
        """
        x_list, y_list = [], []
        for i in range(len(y_train)):
            label = y_train[i]
            ind = self.no_noise_label.index(y_train[i])
            inp = self.no_noise_input[ind]
            for j in range(args.n_aug):
                noisy = self.add_noise(inp)
                x_list.append(noisy)
                y_list.append(label)
        ax_train, ay_train = shuffle(x_list, y_list)
        print('finished augmenting training data!')
        return ax_train, ay_train

    def split_dataset(self, train_ratio):
        """ Splits dataset into train/test sets
            [train_ratio]           Ratio for training; ratio for testing is 1-train_ratio
        """
        n = len(self.labels)
        train_ind = int(np.floor(n*train_ratio))
        x_train = self.inputs[0:train_ind]
        y_train = self.labels[0:train_ind]
        x_test = self.inputs[train_ind:n]
        y_test = self.labels[train_ind:n]
        return x_train, y_train, x_test, y_test

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
    def __init__(self, x, y, args):
        # network architecture
        h1 = tf.nn.relu(fclayer(x,'h1',args.n1))
        self.pred = fclayer(h1,'regression',1)
        self.loss = tf.reduce_mean(tf.square(tf.subtract(y, self.pred)))
        tf.summary.scalar("loss", self.loss)                                        # record scalar for tensorboard log
        self.train_op = tf.train.AdamOptimizer(args.lr).minimize(self.loss)
        # calculate accuracy
        difference = tf.abs(self.pred - y)
        correct = tf.round(difference)
        correct_prediction = tf.equal(correct, tf.zeros(shape=tf.shape(correct)))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", self.accuracy)                                # record scalar for tensorboard log
        self.saver = tf.train.Saver(max_to_keep=1)
        self.merge = tf.summary.merge_all()

def run(sess, x_train, y_train, x_test, y_test, args, case_str):
    """ Trains model for one CV fold and returns lowest loss and accuracy for the fold
        [sess]              Tensorflow session
        [x_train]           Input array
        [y_train]           Labels
        [x_test]            Input array
        [y_test]            Labels
        [args]              Arguments from parser
        [case_str]          String of case# for folder-naming
        """
    low_loss = 1000
    _, input_dim = x_train.shape

    # placeholder for input and labels
    x = tf.placeholder(tf.float32,[None,input_dim])
    y = tf.placeholder(tf.float32)

    # call model and initialize graph
    model = neural_network(x, y, args)
    sess.run(tf.global_variables_initializer())

    # save tensorboard logs
    train_writer = tf.summary.FileWriter( './modelselection'+case_str+'/logs/train', sess.graph)
    test_writer = tf.summary.FileWriter( './modelselection'+case_str+'/logs/test', sess.graph)

    # track training time and test time
    traintime = 0
    time_test = []

    # iterate through epoch
    for i in range(args.n_epochs):
        n_batches, x_batch, y_batch = batching(x_train, y_train, args.batch_size)
        x_batch, y_batch = shuffle(x_batch, y_batch)
        pointer = 0                                                                 # point to right batch
        tracker = i+1                                                               # tracking tf tensorboard log epoch

        # iterate through batches
        for j in range(n_batches):
            xll_train, yll_train = return_batch(x_batch, y_batch, pointer)
            yll_train = np.reshape(yll_train, (-1,1))
            start_train = time.clock()
            _, train_loss, train_pred, summary = sess.run([model.train_op, model.loss, model.pred, model.merge], feed_dict={x: xll_train, y: yll_train})
            end_train = time.clock()
            traintime = traintime + end_train - start_train
            pointer = pointer + 1

        # every fifth epoch, evaluate model
        if i % 5 == 0:
            y_test = np.reshape(y_test, (-1,1))
            start_test = time.clock()
            test_loss, test_pred, summary1, accuracy = sess.run([model.loss, model.pred, model.merge, model.accuracy], feed_dict={x: x_test, y: y_test})
            end_test = time.clock()
            time_test.append(end_test-start_test)
            print('epoch'+str(i))
            print(train_loss)
            print(test_loss)
            print(accuracy)

            # save model if test loss is lower
            if low_loss > test_loss:
                index = sess.run(tf.argmax(tf.abs(y_test-test_pred)))
                low_loss = test_loss
                low_epoch = i
                low_accuracy = accuracy
                low_test_pred, low_y = test_pred[index], y_test[index]
                savePath = model.saver.save(sess, 'modelselection'+case_str+'/checkpoint/my_model.ckpt')
                print('checkpoint saved....................................')

        # write tensorboard logs
        train_writer.add_summary(summary, tracker)
        test_writer.add_summary(summary1, tracker)
        av_testtime = sum(time_test)/len(time_test)

    return low_loss, low_accuracy, low_epoch, low_test_pred, low_y, traintime, av_testtime

if __name__ == '__main__':
    # define parser arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='case1', help='csv filename without the .csv')
    parser.add_argument('--noise', action='store_true', help='adds noise')
    parser.add_argument('--n_aug', type=int, default=200, help='number to augment per training example')
    parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--n1', type=int, default=45, help='number of hidden neurons')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    args = parser.parse_args()

    data = dataset(args)

    # unique folder name for each case
    case_str = args.csv
    if args.noise:
        case_str = 'case'+str(int(case_str[4])+3)

    with tf.Session() as sess:
        # split data into train and test set
        train_ratio = 0.8
        x_train, y_train, x_test, y_test = data.split_dataset(train_ratio)
        # augment data if noise is True
        if args.noise:
            x_train, y_train = data.training_augment(y_train, args)
        # train ANN and return evaluation results
        loss, accuracy, epoch, t_pred, t_label, traintime, av_testtime = run(sess, x_train, y_train, x_test, y_test, args, case_str)
        sess.close()

    # save parameter settings and evaluated metrics
    with open('modelselection'+case_str+'/evaluation.csv', 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['noise', args.noise])
        writer.writerow(['aug_num', args.n_aug])
        writer.writerow(['n1', args.n1])
        writer.writerow(['num_epochs', args.n_epochs])
        writer.writerow(['batch_size', args.batch_size])
        writer.writerow(['loss', loss])
        writer.writerow(['accuracy', accuracy])
        writer.writerow(['epoch', epoch])
        writer.writerow(['pred', t_pred])
        writer.writerow(['label', t_label])
        writer.writerow(['traintime', traintime])
        writer.writerow(['average_testtime', av_testtime])
