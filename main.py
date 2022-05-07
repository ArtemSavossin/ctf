import pandas as pd
import numpy as np
import math
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.datasets import make_blobs
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from random import randrange
from sklearn.datasets import fetch_openml
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.datasets import load_iris
import tensorflow as tf

from sklearn.datasets import fetch_openml

# Fetch the dataset
dataset = fetch_openml("USPS")
print("Dataset USPS loaded...")
data = dataset.data
target = dataset.target # Labels between 0 and 9 to match digits
n_samples = data.shape[0] # Number of samples in the dataset
n_clusters = 10 # Number of clusters to obtain

# Get the split between training/test set and validation set

# Auto-encoder architecture
input_size = data.shape[1]
hidden_1_size = 128
hidden_2_size = 128
hidden_3_size = 500
embedding_size = n_clusters
dimensions = [hidden_1_size, hidden_2_size, hidden_3_size, embedding_size, # Encoder layer dimensions
              hidden_3_size, hidden_2_size, hidden_1_size, input_size] # Decoder layer dimensions
activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, None, # Encoder layer activations
               tf.nn.relu, tf.nn.relu, tf.nn.relu, None] # Decoder layer activations
names = ['enc_hidden_1', 'enc_hidden_2', 'enc_hidden_3', 'embedding', # Encoder layer names
         'dec_hidden_1', 'dec_hidden_2', 'dec_hidden_3', 'output'] # Decoder layer names



ololo = []

def fc_layers(input, specs):
    [dimensions, activations, names] = specs
    for dimension, activation, name in zip(dimensions, activations, names):
        input = tf.layers.dense(inputs=input, units=dimension, activation=activation, name=name, reuse=tf.AUTO_REUSE)
    return input

def autoencoder(input, specs):
    [dimensions, activations, names] = specs
    mid_ind = int(len(dimensions)/2)

    # Encoder
    embedding = fc_layers(input, [dimensions[:mid_ind], activations[:mid_ind], names[:mid_ind]])
    # Decoder
    output = fc_layers(embedding, [dimensions[mid_ind:], activations[mid_ind:], names[mid_ind:]])

    return embedding, output

def f_func(x, y):
    # print('f_func x', x)
    # print('f_func y', y)
    # print('f_func res', tf.reduce_sum(tf.square(x - y), axis=1))
    return tf.reduce_sum(tf.square(x - y), axis=1)

def g_func(x, y):
    # print('g_func x', x)
    # print('g_func y', y)
    # print('g_func res', tf.reduce_sum(tf.square(x - y), axis=1))
    return tf.reduce_sum(tf.square(x - y), axis=1)

class DkmCompGraph(object):
    """Computation graph for Deep K-Means
    """

    def __init__(self, ae_specs, n_clusters, val_lambda):
        input_size = ae_specs[0][-1]
        embedding_size = ae_specs[0][int((len(ae_specs[0])-1)/2)]

        # Placeholder tensor for input data
        self.input = tf.compat.v1.placeholder(dtype=TF_FLOAT_TYPE, shape=(None, input_size))

        # Auto-encoder loss computations
        self.embedding, self.output = autoencoder(self.input, ae_specs)  # Get the auto-encoder's embedding and output
        rec_error = g_func(self.input, self.output)  # Reconstruction error based on distance g

        # k-Means loss computations
        ## Tensor for cluster representatives
        minval_rep, maxval_rep = -1, 1
        self.cluster_rep = tf.Variable(tf.random_uniform([n_clusters, embedding_size],
                                                    minval=minval_rep, maxval=maxval_rep,
                                                    dtype=TF_FLOAT_TYPE), name='cluster_rep', dtype=TF_FLOAT_TYPE)

        ## First, compute the distance f between the embedding and each cluster representative
        list_dist = []
        for i in range(0, n_clusters):
            dist = f_func(self.embedding, tf.reshape(self.cluster_rep[i, :], (1, embedding_size)))
            list_dist.append(dist)
        print('list dist is', tf.stack(list_dist))
        self.stack_dist = tf.stack(list_dist)

        ## Second, find the minimum squared distance for softmax normalization
        min_dist = tf.reduce_min(list_dist, axis=0)

        ## Third, compute exponentials shifted with min_dist to avoid underflow (0/0) issues in softmaxes
        self.alpha = tf.placeholder(dtype=TF_FLOAT_TYPE, shape=())  # Placeholder tensor for alpha
        list_exp = []
        for i in range(n_clusters):
            exp = tf.exp(-self.alpha * (self.stack_dist[i] - min_dist))
            list_exp.append(exp)
        stack_exp = tf.stack(list_exp)
        sum_exponentials = tf.reduce_sum(stack_exp, axis=0)

        ## Fourth, compute softmaxes and the embedding/representative distances weighted by softmax
        list_softmax = []
        list_weighted_dist = []
        for j in range(n_clusters):
            softmax = stack_exp[j] / sum_exponentials
            weighted_dist = self.stack_dist[j] * softmax
            list_softmax.append(softmax)
            list_weighted_dist.append(weighted_dist)
        stack_weighted_dist = tf.stack(list_weighted_dist)

        # Compute the full loss combining the reconstruction error and k-means term
        self.ae_loss = tf.reduce_mean(rec_error)
        self.kmeans_loss = tf.reduce_mean(tf.reduce_sum(stack_weighted_dist, axis=0))
        self.loss = self.ae_loss + val_lambda * self.kmeans_loss

        # The optimizer is defined to minimize this loss
        optimizer = tf.train.AdamOptimizer()
        self.pretrain_op = optimizer.minimize(self.ae_loss) # Pretrain the autoencoder before starting DKM
        self.train_op = optimizer.minimize(self.loss) # Train the whole DKM model



TF_FLOAT_TYPE = tf.float32

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed.
    (Taken from https://github.com/XifengGuo/IDEC-toy/blob/master/DEC.py)
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w) # Optimal label mapping based on the Hungarian algorithm

    summary = []
    for i in ind[0]:
      for j in ind[1]:
        summary.append(w[i, j])
    return sum(summary) * 1.0 / y_pred.size

def next_batch(num, data):
    """
    Return a total of `num` random samples.
    """
    indices = np.arange(1, data.shape[0])
    np.random.shuffle(indices)
    indices = indices[:num]
    batch_data = np.asarray([data[i-1:i].values[0] for i in indices])

    return indices, batch_data


def shuffle(data, target):
    """
    Return a random permutation of the data.
    """
    indices = np.arange(0, len(data))
    np.random.shuffle(indices)
    shuffled_data = np.asarray([data[i] for i in indices])
    shuffled_labels = np.asarray([target[i] for i in indices])

    return shuffled_data, shuffled_labels, indices



import os
import math
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans


# Parameter setting from arguments
n_pretrain_epochs = 50
n_finetuning_epochs = 5
lambda_ = 1.0
batch_size = 128 # Size of the mini-batches used in the stochastic optimizer
n_batches = int(math.ceil(n_samples / batch_size)) # Number of mini-batches
annealing = True # Specify if annealing should be used
seeded = False # Specify if runs are seeded

print("Hyperparameters...")
print("lambda =", lambda_)


constant_value = 1  # specs.embedding_size # Used to modify the range of the alpha scheme
max_n = 20  # Number of alpha values to consider
alphas = np.zeros(max_n, dtype=float)
alphas[0] = 0.1
for i in range(1, max_n):
    alphas[i] = (2 ** (1 / (np.log(i + 1)) ** 2)) * alphas[i - 1]


# Dataset on which the computation graph will be run
data = data

list_acc = []
list_ari = []
list_nmi = []

n_runs = 1
distances = np.zeros((n_clusters, n_samples))
all_indexes = []
for run in range(n_runs):
    print("Run", run)

    # Define the computation graph for DKM
    cg = DkmCompGraph([dimensions, activations, names], n_clusters, lambda_)

    # Run the computation graph
    with tf.Session() as sess:
        # Initialization
        init = tf.global_variables_initializer()
        sess.run(init)

        # Variables to save tensor content

        # Train the full DKM model
        if (len(alphas) > 0):
            print("Starting DKM training...")
        ## Loop over alpha (inverse temperature), from small to large values
        for k in range(len(alphas)):
            print("Training step: alpha[{}]: {}".format(k, alphas[k]))
            cntrs = []
            # Loop over epochs per alpha
            for _ in range(n_finetuning_epochs):
                # Loop over the samples
                for _ in range(n_batches):
                    #print("Training step: alpha[{}], epoch {}".format(k, i))

                    # Fetch a random data batch of the specified size
                    indices, data_batch = next_batch(batch_size, data)
                    all_indexes.append(indices)
                    #print(tf.trainable_variables())
                    #current_batch_size = np.shape(data_batch)[0] # Can be different from batch_size for unequal splits

                    # Run the computation graph on the data batch
                    _, loss_, stack_dist_, cluster_rep_, ae_loss_, kmeans_loss_ =\
                        sess.run((cg.train_op, cg.loss, cg.stack_dist, cg.cluster_rep, cg.ae_loss, cg.kmeans_loss),
                                 feed_dict={cg.input: data_batch, cg.alpha: alphas[k]})
                    cntrs = cluster_rep_
                    # Save the distances for batch samples
                    for j in range(len(indices)):
                        distances[:, indices[j]] = stack_dist_[:, j]
            print('are centroids chanhing?', cntrs)
            # Evaluate the clustering performance every print_val alpha and for last alpha
            print_val = 1
            if k % print_val == 0 or k == max_n - 1:
                # print("loss:", loss_)
                # print("ae loss:", ae_loss_)
                print("kmeans loss:", kmeans_loss_)

                # Infer cluster assignments for all samples
                cluster_assign = np.zeros((n_samples), dtype=float)
                for i in range(n_samples):
                    # print('popa jopa', distances[:, i])
                    # print('n min', np.min(distances[:, i]))
                    index_closest_cluster = np.argmin(distances[:, i])
                    cluster_assign[i] = index_closest_cluster

                print('is anything changing?', distances[:, 13])
                cluster_assign = cluster_assign.astype(np.int64)
                ttarget = []
                for i in range(len(target.values)):
                    ttarget.append(int(target.values[i]))
                assignn = []
                for i in range(len(cluster_assign)):
                    assignn.append(int(cluster_assign[i]) + 1)
                ololo.append(assignn)
                # Evaluate the clustering performance using the ground-truth labels
                # acc = cluster_acc(target, cluster_assign)
                # print("ACC", acc)
                ari = adjusted_rand_score(ttarget, assignn)
                print("ARI", ari)
                # nmi = normalized_mutual_info_score(ttarget, assignn)
                # print("NMI", nmi)
        # list_acc.append(acc)
        list_ari.append(ari)
        # list_nmi.append(nmi)



# list_acc = np.array(list_acc)
# print("Average ACC: {:.3f} +/- {:.3f}".format(np.mean(list_acc), np.std(list_acc)))
list_ari = np.array(list_ari)
print("Average ARI: {:.3f} +/- {:.3f}".format(np.mean(list_ari), np.std(list_ari)))
# list_nmi = np.array(list_nmi)
# print("Average NMI: {:.3f} +/- {:.3f}".format(np.mean(list_nmi), np.std(list_nmi)))

