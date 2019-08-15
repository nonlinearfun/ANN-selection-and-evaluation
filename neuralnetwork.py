import numpy as np
import tensorflow as tf
import pandas
import time
import csv
import matplotlib.pyplot as plt

class dataset:
    def __init__(self, args):
        self.inputs, self.labels = None, None
        self.x_train_list, self.y_train_list, self.x_test_list, self.y_test_list = [],[],[],[]
        self.process_data(args)

    def process_data(self, args):
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

def add_noise(inputs, labels, aug_num):
    """ Create and then add gaussian noise - zero mean and std of 0.1*||inp||_infty
        [inputs]           array that needs noise
        [labels]           labels of inputs
        [aug_num]          # for data augmentation
    """
    inf_norm = (abs(inputs)).max(axis=1)
    _, d = inputs.shape
    std_v = np.transpose(np.tile(0.1*inf_norm, (d,1)))
    inp = np.repeat(inputs, aug_num, axis=0)
    std_v = np.repeat(std_v, aug_num, axis=0)
    labels = np.repeat(labels, aug_num, axis=0)
    noise = np.random.normal(loc=0,scale=std_v,size=inp.shape)
    return inp+noise, labels

def split_dataset(inputs, labels, train_ratio):
    """ Splits dataset into train/test sets
        [train_ratio]      ratio for training; ratio for testing is 1-train_ratio
    """
    n = len(labels)
    train_ind = int(np.floor(n*train_ratio))
    x_train = inputs[0:train_ind]
    y_train = labels[0:train_ind]
    x_test = inputs[train_ind:n]
    y_test = labels[train_ind:n]
    return x_train, y_train, x_test, y_test

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

def run(sess, X_train, y_train, X_test, y_test, args, case_str):
    """ Trains model for one CV fold and returns lowest loss and accuracy for the fold
        [sess]              Tensorflow session
        [x_train]           Input array
        [y_train]           Labels
        [x_test]            Input array
        [y_test]            Labels
        [args]              Arguments from parser
        [case_str]          String of case# for folder-naming
        """
    low_loss = 10000

    n_train, input_dim = X_train.shape
    n_test, _ = X_test.shape
    n_batch = int(np.ceil(n_train/args.bs))

    x = tf.placeholder(tf.float32, shape=[None, input_dim])
    y = tf.placeholder(tf.float32, shape=[None, 1])

    # Tensorflow's Dataset pipeline with reinitializable iterator: https://www.tensorflow.org/guide/datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x,y)).shuffle(100).batch(args.bs).repeat()
    test_dataset = tf.data.Dataset.from_tensor_slices((x,y)).batch(n_test)
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    features, labels = iterator.get_next()
    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    # initialize model
    model = neural_network(features,labels,args)
    sess.run(tf.global_variables_initializer())

    # save tensorboard logs
    train_writer = tf.summary.FileWriter( './modelselection'+case_str+'/logs/train', sess.graph)
    test_writer = tf.summary.FileWriter( './modelselection'+case_str+'/logs/test', sess.graph)

    # track training time and test time
    traintime = 0
    time_test = []

    for i in range(args.n_epochs):
        tracker = i+1
        sess.run(train_init_op, feed_dict = {x : X_train, y: y_train})
        loss_total = 0
        for j in range(n_batch):
            start_train = time.clock()
            _, train_loss, train_acc, summary = sess.run([model.train_op, model.loss, model.accuracy, model.merge])
            end_train = time.clock()
            traintime = traintime + end_train - start_train
            loss_total += train_loss

        sess.run(test_init_op, feed_dict = {x : X_test, y: y_test})
        start_test = time.clock()
        test_loss, test_acc, test_pred, summary1 = sess.run([model.loss, model.accuracy, model.pred, model.merge])
        end_test = time.clock()
        time_test.append(end_test-start_test)

        print('epoch'+str(i))
        print(train_loss)
        print(test_loss)
        print(test_acc)

        if low_loss > test_loss:
            index = sess.run(tf.argmax(tf.abs(y_test-test_pred)))
            low_loss = test_loss
            low_epoch = i
            low_accuracy = test_acc
            low_test_pred, low_y = test_pred[index], y_test[index]
            savePath = model.saver.save(sess, 'modelselection'+case_str+'/checkpoint/my_model.ckpt')
            print('checkpoint saved....................................')

        # write tensorboard logs
        train_writer.add_summary(summary, tracker)
        test_writer.add_summary(summary1, tracker)
        av_testtime = sum(time_test)/len(time_test)

    return low_loss, low_accuracy, low_epoch, low_test_pred, low_y, traintime, av_testtime


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='case1', help='csv filename without the .csv')
    parser.add_argument('--noise', action='store_true', help='adds noise')
    parser.add_argument('--n_aug', type=int, default=100, help='number to augment per training example')
    parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--n1', type=int, default=45, help='number of hidden neurons')
    parser.add_argument('--bs', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.001, help='gamma')
    args = parser.parse_args()

    data = dataset(args)
    X, y = shuffle(data.inputs, data.labels)
    train_ratio = 0.8
    x_train, y_train, x_test, y_test = split_dataset(X, y, train_ratio)

    # unique folder name for each case
    case_str = args.csv
    if args.noise:
        case_str = 'case'+str(int(case_str[4])+3)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # add noise and augment training if noise is True
        if args.noise:
            x_train, y_train = add_noise(x_train, y_train, args.n_aug)
            x_test, y_test = add_noise(x_test, y_test, 1)

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
        writer.writerow(['batch_size', args.bs])
        writer.writerow(['gamma', args.gamma])
        writer.writerow(['loss', loss])
        writer.writerow(['accuracy', accuracy])
        writer.writerow(['epoch', epoch])
        writer.writerow(['pred', t_pred])
        writer.writerow(['label', t_label])
        writer.writerow(['traintime', traintime])
        writer.writerow(['average_testtime', av_testtime])
