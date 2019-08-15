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
import contextlib
import os
import matplotlib.pyplot as plt
import neuralnetwork_tf2
from neuralnetwork_tf2 import *
import time
import multiprocessing
from multiprocessing import Process
from sklearn.model_selection import ParameterGrid

np.random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@contextlib.contextmanager
def temp_seed(seed):
    """set numpy seed to keep consistency across all models (to allow for comparision)
        1) save the original global random state
        2) set seed, so that np.random return same numbers as next call to same temp_seed
        3) reinstate global random state
        Result: Batch splits and shuffling kept same across models
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def fit(data, n1, lr, bs, bayes_scope, model_num, args):
    """ Return mean and std of loss and accuracy across folds in a single model
        [data]                  dataset
        [n1],[lr],[bs]          hyperparameters (see below)
        [bayes_scope]           help create new variable names in tf graph
        [model_num]             track model number
    """
    tf.set_random_seed(1234)            # set tf seed to keep same weight initialization for each model
    n_epochs = args.n_epochs            # number of epochs

    # track losses and accuracies of each fold
    loss = []
    accuracy = []
    test_list = []

    # perform cross validation
    with temp_seed(1234):
        for counter in range(len(data.y_train_list)):
            bayes_scope += 1
            with tf.Session() as sess:
                x_train, y_train, x_test, y_test = data.return_cross_val(counter)
                # add noise to inputs
                if args.noise:
                    x_train, y_train = data.add_noise(x_train, y_train, args.n_aug)
                    x_test, y_test = data.add_noise(x_test, y_test, 1)
                low_loss, low_accuracy = run(sess, x_train, y_train, x_test, y_test, n1, lr, bs, n_epochs, str(bayes_scope), model_num)
                loss.append(low_loss)
                accuracy.append(low_accuracy)
                sess.close()
            tf.reset_default_graph()
            test_list.append(y_test)

    # calculate mean and std of losses and accuracies across the folds
    loss = np.array(loss)
    accuracy = np.array(accuracy)
    loss_mean = np.mean(loss)
    loss_std = np.std(loss)
    accuracy_mean = np.mean(accuracy)
    accuracy_std = np.std(accuracy)

    return loss, accuracy, loss_mean, loss_std, accuracy_mean, accuracy_std, test_list

def model_selection(data, grid, args, work_id, model_num):
    """ Determine the best hyperparameter settings via a grid search-like procedure
        [data]                  dataset
        [grid]                  hyperparameter grid
        [bayes_scope]           help create new variable names in tf graph
        [work_id]               process number
        [model_num]             track model number
        [args]                  arguments from parser
    """
    low_score = 1000
    low_model_num = 0
    low_n1 = 0
    low_bs = 0
    low_lr = 0
    bayes_scope = 0

    # create folder if folder does not exist!
    case_str = args.csv
    if args.noise:
        case_str = 'case'+str(int(case_str[4])+3)

    if not os.path.exists('modelselection'+case_str):
        os.makedirs('modelselection'+case_str)

    # open csv file to store information
    csv_filename = 'modelselection'+case_str+'/thread_id'+work_id+'.csv'
    with open(csv_filename, 'a') as csv_file:
        writer = csv.writer(csv_file)

        # iterate through all combinations of hyperparameter settings
        for params in grid:
            model_num += 1
            loss, accuracy, loss_mean, loss_std, accuracy_mean, accuracy_std, test_list = fit(data, params['n1'], params['lr'], params['bs'], bayes_scope, model_num, args)

            # write all hyperparameters and scores into csv
            writer.writerow(['model_num', model_num])
            writer.writerow(['n1', params['n1']])
            writer.writerow(['lr', params['lr']])
            writer.writerow(['bs', params['bs']])
            writer.writerow(['loss', loss])
            writer.writerow(['loss_mean', loss_mean])
            writer.writerow(['loss_std', loss_std])
            writer.writerow(['accuracy', accuracy])
            writer.writerow(['accuracy_mean', accuracy_mean])
            writer.writerow(['accuracy_std', accuracy_std])
            writer.writerow(['test_list', test_list])

def worker(data, grid, args, work_id, model_num):
    """ Have process perform search instances """
    model_selection(data, grid, args, work_id, model_num)
    return

if __name__ == '__main__':
    # define parser arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='case1', help='csv filename without the .csv')
    parser.add_argument('--noise', action='store_true', help='adds noise')
    parser.add_argument('--n_aug', type=int, default=100, help='number to augment per training example')
    parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs')
    args = parser.parse_args()

    # specify and construct grid for grid search
    if args.noise:
        n1_v = np.arange(70,120,5)                       # num hidden neurons
        lr_v = np.array([1e-3,5e-4,1e-4])                # learning rate
        bs_v = np.array([32,64])                         # batch size
    else:
        n1_v = np.arange(180,205,5)                      # num hidden neurons
        lr_v = np.array([1e-3,5e-4,1e-4])                # learning rate
        bs_v = np.array([4,8,16])                        # batch size

    combo_array = {'n1': n1_v, 'lr': lr_v, 'bs': bs_v}
    grid = ParameterGrid(combo_array)
    data = dataset(args)

    # use multiprocessing module to run grid search in parallel with independent processes
    num_model = len(list(grid))
    print('NUMBER OF TOTAL RUNS:', num_model)
    num_cpu = multiprocessing.cpu_count()
    percpu = int(np.ceil(num_model/num_cpu))

    for i in range(num_cpu):
        print(i)
        if i == num_cpu - 1:
            p = multiprocessing.Process(target=worker, args=(data, list(grid)[i*percpu:], args, str(i), i*percpu))
        else:
            p = multiprocessing.Process(target=worker, args=(data, list(grid)[i*percpu:i*percpu+percpu], args, str(i), i*percpu))
        p.start()
    p.join()
