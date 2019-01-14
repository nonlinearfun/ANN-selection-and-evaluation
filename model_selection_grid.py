import numpy as np
import contextlib
import os
import matplotlib.pyplot as plt
import neuralnetwork_tf_cpu
from neuralnetwork_tf_cpu import *

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
    with temp_seed(0):
        for counter in range(len(data.y_train_list)):
            bayes_scope += 1
            with tf.Session() as sess:
                x_train, y_train, x_test, y_test = data.return_cross_val(counter)
                if args.noise:
                    x_train, y_train = data.training_augment(y_train, args.n_aug)
                low_loss, low_accuracy = run(sess, x_train, y_train, x_test, y_test, n1, lr, bs, n_epochs, str(bayes_scope), counter, model_num)
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

def model_selection(data, h1_v, lr_v, bs_v, args):
    """ Determine the best hyperparameter settings via a grid search-like procedure
        [data]                  dataset
        [n1_v],[lr_v],[bs_v]    hyperparameter array (see below)
        [bayes_scope]           help create new variable names in tf graph
        [model_num]             track model number
        [args]                  arguments from parser
    """
    low_score = 1000
    bayes_scope = 0
    model_num = 0

    # create folder if folder does not exist!
    case_str = args.csv
    if args.noise:
        case_str = 'case'+str(int(case_str[4])+3)

    if not os.path.exists('modelselection'+case_str):
        os.makedirs('modelselection'+case_str)

    # open csv file to store information
    csv_filename = 'modelselection'+case_str+'/hyperpandlosses'+case_str+'.csv'
    with open(csv_filename, 'a') as csv_file:
        writer = csv.writer(csv_file)

        # iterate through all combinations of hyperparameter settings
        for a in range(len(h1_v)):
            n1 = h1_v[a]
            for b in range(len(lr_v)):
                 lr = lr_v[b]
                 for c in range(len(bs_v)):
                     bs = bs_v[c]
                     loss, accuracy, loss_mean, loss_std, accuracy_mean, accuracy_std, test_list = fit(data, n1, lr, bs, bayes_scope, model_num, args)
                     model_num += 1

                     # write all hyperparameters and scores into csv
                     writer.writerow(['model_num', model_num])
                     writer.writerow(['n1', n1])
                     writer.writerow(['lr', lr])
                     writer.writerow(['bs', bs])
                     writer.writerow(['loss', loss])
                     writer.writerow(['loss_mean', loss_mean])
                     writer.writerow(['loss_std', loss_std])
                     writer.writerow(['accuracy', accuracy])
                     writer.writerow(['accuracy_mean', accuracy_mean])
                     writer.writerow(['accuracy_std', accuracy_std])
                     writer.writerow(['test_list', test_list])

                     # if better score, keep track of hyperparameters
                     if loss_mean < low_score:
                         low_n1 = n1
                         low_lr = lr
                         low_bs = bs
                         low_model_num = model_num
                         low_score = loss_mean
        writer.writerow(['low_model_num', low_model_num])
    print('best hyperparameters are n1=',low_n1,', lr=',low_lr,', bs=',low_bs, ': model #',low_model_num)

if __name__ == '__main__':
    # define parser arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='case1', help='csv filename without the .csv')
    parser.add_argument('--noise', action='store_true', help='adds noise')
    parser.add_argument('--n_aug', type=int, default=200, help='number to augment per training example')
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs')
    args = parser.parse_args()

    # combinations of hyperparameters to select
    if args.noise:
        h1_v = np.arange(20,51,10)                      # num hidden neurons
        lr_v = np.array([1e-4,5e-4,5e-5])               # learning rate
        bs_v = np.array([8,16,32,64])                   # batch size
    else:
        h1_v = np.arange(20,61,5)                       # num hidden neurons
        lr_v = np.array([1e-4,5e-4,5e-5,1e-5])          # learning rate
        bs_v = np.arange(4,9,2)                         # batch size

    data = dataset(args)                            # initialize dataset class here, so that CV folds kept same across models
    model_selection(data, h1_v, lr_v, bs_v, args)   # perform model selection
