import numpy as np
import tensorflow as tf
import pandas
import time
import csv
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class dataset:
    def __init__(self, args):
        self.inputs, self.labels = None, None
        self.x_train_list, self.y_train_list, self.x_test_list, self.y_test_list = [],[],[],[]
        self.process_data(args)

    def process_data(self, args):
        """ Load data and create cross validation (CV) splits """
        try:
            dataset = np.load('data'+args.csv+'.npz')
        except:
            df = pandas.read_csv(args.csv+'.csv')
            data = df.values
            m,n = data.shape
            inputs, labels = [], []
            header = list(df)
            for ind, val in enumerate(header):
                inp = np.float32(data[:,ind])
                label = np.float32(val[1:].replace('_','.'))
                inputs.append(inp)
                labels.append(label)
            np.savez('data'+args.csv+'.npz', X=inputs, y=labels)
            dataset = np.load('data'+args.csv+'.npz')
        self.inputs = dataset['X']
        self.labels = np.reshape(dataset['y'], [-1, 1])
        self.cross_val()

    def add_noise(self, inputs, labels, aug_num):
        """ Create and then add gaussian noise - zero mean and std of 0.1*||inp||_infty
            [inputs]           array that needs noise
        """
        inf_norm = (abs(inputs)).max(axis=1)
        _, d = inputs.shape
        std_v = np.transpose(np.tile(0.1*inf_norm, (d,1)))
        inp = np.repeat(inputs, aug_num, axis=0)
        std_v = np.repeat(std_v, aug_num, axis=0)
        labels = np.repeat(labels, aug_num, axis=0)
        noise = np.random.normal(loc=0,scale=std_v,size=inp.shape)
        return inp+noise, labels

    def cross_val(self):
        """Create list with CV splits"""
        kf = KFold(n_splits=5, shuffle=True)
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
        difference = tf.abs(tf.squeeze(self.pred) - tf.squeeze(y))
        correct = tf.round(difference)
        correct_prediction = tf.equal(correct, tf.zeros(shape=tf.shape(correct)))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def run(sess, X_train, y_train, X_test, y_test, n1, lr, bs, n_epochs, bayes_scope, model_num):
    """ Trains model for one CV fold and returns lowest loss and accuracy for the fold
        [sess]              Tensorflow session
        [X_train]           Input array
        [y_train]           Labels
        [X_test]            Input array
        [y_test]            Labels
        [n1],[lr],[bs]      Hyperparameters
        [n_epochs]          Number of epochs
        [bayes_scope]       To help create new graph
        [model_num]         Model number
    """
    low_loss = 10000
    tf.set_random_seed(1234)

    n_train, input_dim = X_train.shape      # number and dimensionality of training points
    n_test, _ = X_test.shape                # number of test points
    n_batch = int(np.ceil(n_train/bs))      # number of batchs

    x = tf.placeholder(tf.float32, shape=[None, input_dim])
    y = tf.placeholder(tf.float32, shape=[None, 1])

    # Tensorflow's Dataset pipeline with reinitializable iterator: https://www.tensorflow.org/guide/datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x,y)).shuffle(100).batch(bs).prefetch(1).repeat()
    test_dataset = tf.data.Dataset.from_tensor_slices((x,y)).batch(n_test).prefetch(1)
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    features, labels = iterator.get_next()
    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    model = neural_network(features,labels,n1,lr,bayes_scope)
    sess.run(tf.global_variables_initializer())

    for i in range(n_epochs):
        sess.run(train_init_op, feed_dict = {x : X_train, y: y_train})
        loss_total = 0
        for j in range(n_batch):
            _, train_loss = sess.run([model.train_op, model.loss])
            loss_total += train_loss

        # evaluate test
        sess.run(test_init_op, feed_dict = {x : X_test, y: y_test})
        test_loss, test_acc = sess.run([model.loss, model.accuracy])
        # print('epoch'+str(i))
        # print('train_loss:'+str(loss_total/bs))
        # print('test_loss:'+str(test_loss))
        # print('test_acc:'+str(test_acc))
        if low_loss > test_loss:
            low_loss = test_loss
            low_accuracy = test_acc
    return low_loss, low_accuracy
