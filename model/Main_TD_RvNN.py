# -*- coding: utf-8 -*-
"""
@object: Twitter
@task: Main function of recursive NN (4 classes)
@author: majing
@structure: Top-Down recursive Neural Networks
@variable: Nepoch, lr, obj, fold
@time: Jan 24, 2018
"""

import sys
import TD_RvNN
import numpy as np
import time
import datetime
from evaluate import *
from typing import List


# tools
def str2matrix(_str, max_l):  # str = index:word_freq index:word_freq
    word_freq, word_index = [], []
    l = 0
    for pair in _str.split(' '):
        word_freq.append(float(pair.split(':')[1]))
        word_index.append(int(pair.split(':')[0]))
        l += 1
    ladd = [0 for i in range(max_l - l)]
    word_freq += ladd
    word_index += ladd
    return word_freq, word_index


def load_label(label, l1, l2, l3, l4):
    label_set_nr, label_set_f, label_set_t, label_set_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
    _y_train = None
    if label in label_set_nr:
        _y_train = [1, 0, 0, 0]
        l1 += 1
    if label in label_set_f:
        _y_train = [0, 1, 0, 0]
        l2 += 1
    if label in label_set_t:
        _y_train = [0, 0, 1, 0]
        l3 += 1
    if label in label_set_u:
        _y_train = [0, 0, 0, 1]
        l4 += 1
    return _y_train, l1, l2, l3, l4


def construct_tree(tree):
    # tree: {index1:{'parent':, 'maxL':, 'vec':}

    # 1. ini tree node
    index2node = {}
    for i in tree:
        node = TD_RvNN.NodeTweet(idx=i)
        index2node[i] = node

    # 2. construct tree
    _j = 0
    root = None
    for _j in tree:
        index_c = _j
        index_p = tree[_j]['parent']
        node_c = index2node[index_c]
        word_freq, word_index = str2matrix(tree[_j]['vec'], tree[_j]['maxL'])
        node_c.index = word_index
        node_c.word = word_freq

        # not root node ##
        if not index_p == 'None':
            node_p = index2node[int(index_p)]
            node_c.parent = node_p
            node_p.children.append(node_c)

        # root node ##
        else:
            root = node_c

    # 3. convert tree to DNN input
    parent_num = tree[_j]['parent_num']
    ini_x, ini_index = str2matrix("0:0", tree[_j]['maxL'])
    x_word, x_index, tree = TD_RvNN.gen_nn_inputs(root, ini_x)
    return x_word, x_index, tree, parent_num


# load data
def load_data(_label_path, _tree_path, _train_path, _test_path, _eid_pool: List[str]):
    print("loading tree label", end=' ')
    label_dic = {}
    for line in open(_label_path):
        line = line.rstrip()
        label, eid = line.split('\t')[0], line.split('\t')[2]
        label_dic[eid] = label.lower()
    print(len(label_dic))

    print("reading tree", end=' ')  # X
    tree_dic = {}
    for line in open(_tree_path):
        line = line.rstrip()
        eid, index_p, index_c = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        parent_num, max_l = int(line.split('\t')[3]), int(line.split('\t')[4])
        vec = line.split('\t')[5]
        if eid not in tree_dic:
            tree_dic[eid] = {}
        tree_dic[eid][index_c] = {'parent': index_p, 'parent_num': parent_num, 'maxL': max_l, 'vec': vec}
    print('tree no:', len(tree_dic))

    print("loading train set", end=' ')
    tree_train, word_train, index_train, y_train, parent_num_train, c = [], [], [], [], [], 0
    l1, l2, l3, l4 = 0, 0, 0, 0
    unavailable_train_eids = set()
    for eid in open(_train_path):
        # if c > 8: break
        eid = eid.rstrip()
        if _eid_pool and str(eid) not in _eid_pool:
            unavailable_train_eids.add(eid)
            continue
        if eid not in label_dic:
            continue
        if eid not in tree_dic:
            continue
        if len(tree_dic[eid]) <= 0:
            continue

        # 1. load label
        label = label_dic[eid]
        y, l1, l2, l3, l4 = load_label(label, l1, l2, l3, l4)
        y_train.append(y)

        # 2. construct tree
        x_word, x_index, tree, parent_num = construct_tree(tree_dic[eid])
        tree_train.append(tree)
        word_train.append(x_word)
        index_train.append(x_index)
        parent_num_train.append(parent_num)
        c += 1
    print(l1, l2, l3, l4)

    print("loading test set", end=' ')
    tree_test, word_test, index_test, parent_num_test, y_test, c = [], [], [], [], [], 0
    l1, l2, l3, l4 = 0, 0, 0, 0
    unavailable_test_eids = set()
    for eid in open(_test_path):
        # if c > 4: break
        eid = eid.rstrip()
        if _eid_pool and str(eid) not in _eid_pool:
            unavailable_test_eids.add(eid)
            continue
        if eid not in label_dic:
            continue
        if eid not in tree_dic:
            continue
        if len(tree_dic[eid]) <= 0:
            continue

        # 1. load label
        label = label_dic[eid]
        y, l1, l2, l3, l4 = load_label(label, l1, l2, l3, l4)
        y_test.append(y)

        # 2. construct tree
        x_word, x_index, tree, parent_num = construct_tree(tree_dic[eid])
        tree_test.append(tree)
        word_test.append(x_word)
        index_test.append(x_index)
        parent_num_test.append(parent_num)
        c += 1

    print(l1, l2, l3, l4)
    print("train no:", len(tree_train), len(word_train), len(index_train), len(parent_num_train), len(y_train))
    print("test no:", len(tree_test), len(word_test), len(index_test), len(parent_num_test), len(y_test))
    print("dim1 for 0:", len(tree_train[0]), len(word_train[0]), len(index_train[0]))
    print("case 0:", tree_train[0][0], word_train[0][0], index_train[0][0], parent_num_train[0])
    print('Unavailable eids, train: {}, test: {}'.format(len(unavailable_train_eids), len(unavailable_test_eids)))
    return tree_train, word_train, index_train, parent_num_train, y_train, \
           tree_test, word_test, index_test, parent_num_test, y_test


# MAIN

def run(_vocabulary_size, _hidden_dim, _n_class, _n_epoch, _learning_rate,
        _label_path, _tree_path, _train_path, _test_path, _eid_pool):

    # 1. load tree & word & index & label
    tree_train, word_train, index_train, parent_num_train, y_train, \
    tree_test, word_test, index_test, parent_num_test, y_test = load_data(
        _label_path, _tree_path, _train_path, _test_path, _eid_pool
    )

    # 2. ini RNN model
    t0 = time.time()
    model = TD_RvNN.RvNN(_vocabulary_size, _hidden_dim, _n_class)
    t1 = time.time()
    print('Recursive model established,', (t1 - t0) / 60)

    # if os.path.isfile(modelPath):
    #   load_model_Recursive_gruEmb(modelPath, model)
    #   lr = 0.0001

    # debug here
    # print len(tree_test[121]), len(index_test[121]), len(word_test[121])
    # print tree_test[121]
    # exit(0)
    # loss, pred_y = model.train_step_up(word_test[121], index_test[121], tree_test[121], y_test[121], lr)
    # print loss, pred_y
    # exit(0)
    '''i=568
    loss, pred_y = model.train_step_up(word_train[i], index_train[i], parent_num_train[i], tree_train[i], y_train[i], lr)
    print loss, pred_y
    print len(tree_train[i]), len(word_train[i]), parent_num_train[i]
    print tree_train[i]
    print word_train[i]
    print 'final_state:',model._evaluate(word_train[i], index_train[i], parent_num_train[i], tree_train[i])
    tree_states=model._evaluate2(word_train[i], index_train[i], parent_num_train[i], tree_train[i])
    print 'tree_states:', tree_states
    print tree_states[-1:].mean(axis=0)
    tree_states_test=model._evaluate3(word_train[i], index_train[i], tree_train[i])
    print 'l:',len(tree_states_test)
    print 'lo:',tree_states_test[parent_num_train[i]:]'''
    #

    # 3. looping SGD
    losses_5, losses = [], []
    num_examples_seen = 0
    for epoch in range(_n_epoch):

        # one SGD
        indexes = [i for i in range(len(y_train))]
        for i in indexes:
            '''print i,":", len(tree_train[i])
            print tree_train[i]
            tree_state = model._state(word_train[i], index_train[i], child_num_train[i], tree_train[i])
            print len(tree_state)
            print tree_state
            evl = model._evaluate(word_train[i], index_train[i], child_num_train[i], tree_train[i])
            print len(evl) 
            print evl'''
            loss, pred_y = model.train_step_up(word_train[i], index_train[i], parent_num_train[i], tree_train[i],
                                               y_train[i], _learning_rate)
            # print loss, pred_y
            losses.append(round(float(loss), 2))
            '''if math.isnan(loss):
            #   continue 
               print loss, pred_y
               print i
               print len(tree_train[i]), len(word_train[i]), parent_num_train[i]
               print tree_train[i]
               print word_train[i]
               print 'final_state:',model._evaluate(word_train[i], index_train[i], parent_num_train[i], tree_train[i])'''
            num_examples_seen += 1
        print("epoch=%d: loss=%f" % (epoch, np.mean(losses)))
        # floss.write(str(time)+": epoch="+str(epoch)+" loss="+str(loss) +'\n')
        sys.stdout.flush()
        # print losses
        # exit(0)

        # cal loss & evaluate
        if epoch % 5 == 0:
            losses_5.append((num_examples_seen, np.mean(losses)))
            time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(
                "%s: Loss after num_examples_seen=%d epoch=%d: %f" %
                (time_now, num_examples_seen, epoch, np.mean(losses))
            )
            # floss.write(str(time)+": epoch="+str(epoch)+" loss="+str(loss) +'\n')
            # floss.flush()
            sys.stdout.flush()
            prediction = []
            for j in range(len(y_test)):
                # print j
                prediction.append(model.predict_up(word_test[j], index_test[j], parent_num_test[j], tree_test[j]))
            res = evaluation_4class(prediction, y_test)
            print('results:', res)
            # floss.write(str(res)+'\n')
            # floss.flush()
            sys.stdout.flush()

            # Adjust the learning rate if loss increases
            if len(losses_5) > 1 and losses_5[-1][1] > losses_5[-2][1]:
                _learning_rate = _learning_rate * 0.5
                print("Setting learning rate to %f" % _learning_rate)
                # floss.write("Setting learning rate to:"+str(lr)+'\n')
                # floss.flush()
                sys.stdout.flush()
            # save_model_Recursive_gruEmb(modelPath, model)
        # floss.flush()
        losses = []

    # floss.close()


def run_wrapper(path_root, _fold, _eid_pool, _hidden_dim=100, _n_epoch=600, _learning_rate=0.005):
    _obj = "Twitter1516"
    _vocabulary_size = 5000

    run(
        _vocabulary_size=_vocabulary_size,
        _hidden_dim=_hidden_dim,
        _n_class=4,
        _n_epoch=_n_epoch,
        _learning_rate=_learning_rate,
        _label_path=path_root + "/resource/" + _obj + "_label_All.txt",
        _tree_path=path_root + '/resource/data.TD_RvNN.vol_' + str(_vocabulary_size) + '.txt',
        _train_path=path_root + "/nfold/RNNtrainSet_" + _obj + str(_fold) + "_tree.txt",
        _test_path=path_root + "/nfold/RNNtestSet_" + _obj + str(_fold) + "_tree.txt",
        _eid_pool=_eid_pool,
    )


if __name__ == '__main__':
    obj = "Twitter1516"  # choose dataset, you can choose either "Twitter15" or "Twitter16"
    fold = "2"  # fold index, choose from 0-4
    tag = "_u2b"
    vocabulary_size = 5000

    unit = "TD_RvNN-" + obj + str(fold) + '-vol.' + str(vocabulary_size) + tag
    # lossPath = "../loss/loss-"+unit+".txt"
    # modelPath = "../param/param-"+unit+".npz"
    # floss = open(lossPath, 'a+')

    run(
        _vocabulary_size=vocabulary_size,
        _hidden_dim=100,
        _n_class=4,
        _n_epoch=600,
        _learning_rate=0.005,
        _label_path="../resource/" + obj + "_label_All.txt",
        _tree_path='../resource/data.TD_RvNN.vol_' + str(vocabulary_size) + '.txt',
        _train_path="../nfold/RNNtrainSet_" + obj + str(fold) + "_tree.txt",
        _test_path="../nfold/RNNtestSet_" + obj + str(fold) + "_tree.txt",
        _eid_pool=[],
    )

