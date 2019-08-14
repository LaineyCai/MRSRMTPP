'''
Author: Hongyun Cai
Date: 2018.3.13
Functionality: read sequences, structure from file
2018.03.23: add tg mask, to distinguish sequence with timegap >5, 0.5-5, 0.001-0.5, <0.001 four ranges, fit different weights
2018.04.02: add multiple relation in graphs
'''
import os
# import re
# import sys
import codecs
import networkx as nx
import numpy as np
import pickle
# import pprint
from theano import config
from collections import defaultdict

def get_range(gap):
    if gap ==0:
        return 0
    elif gap >0:
        return 1

def load_graph(data_dir):
    # loads nodes observed in any cascade.
    node_file = os.path.join(data_dir, 'seen_nodes.txt')
    with open(node_file, 'rb') as f:
        seen_nodes = [x.strip() for x in f]

    # builds node index
    node_index = {v: i for i, v in enumerate(seen_nodes)}

    # loads graph
    parent_file = os.path.join(data_dir, 'parentgraph.txt')
    sibling_file = os.path.join(data_dir, 'siblinggraph.txt')
    self_file = os.path.join(data_dir, 'selfgraph.txt')
    parentpkl_file = os.path.join(data_dir, 'parent.pkl')
    siblingpkl_file = os.path.join(data_dir, 'sibling.pkl')
    selfpkl_file = os.path.join(data_dir, 'self.pkl')

    if os.path.isfile(parentpkl_file):
        Gp = pickle.load(open(parentpkl_file, 'rb'))
        Gs = pickle.load(open(siblingpkl_file, 'rb'))
        Gi = pickle.load(open(selfpkl_file, 'rb'))
    else:
        Gp = nx.Graph()
        Gp.name = data_dir+'parent'
        n_nodes = len(node_index)
        Gp.add_nodes_from(range(n_nodes))
        with open(parent_file, 'rb') as f:
            f.next()
            for line in f:
                u, v = line.strip().split()
                if (u in node_index) and (v in node_index):
                    u = node_index[u]
                    v = node_index[v]
                    Gp.add_edge(u, v)
        f.close()
        pickle.dump(Gp, open(parentpkl_file, 'wb'))
        Gs = nx.Graph()
        Gs.name = data_dir + 'sibling'
        Gs.add_nodes_from(range(n_nodes))
        with open(sibling_file, 'rb') as f:
            f.next()
            for line in f:
                u, v = line.strip().split()
                if (u in node_index) and (v in node_index):
                    u = node_index[u]
                    v = node_index[v]
                    Gs.add_edge(u, v)
        f.close()
        pickle.dump(Gs, open(siblingpkl_file, 'wb'))

        Gi = nx.Graph()
        Gi.name = data_dir + 'self'
        Gi.add_nodes_from(range(n_nodes))
        with open(self_file, 'rb') as f:
            f.next()
            for line in f:
                u, v = line.strip().split()
                if (u in node_index) and (v in node_index):
                    u = node_index[u]
                    v = node_index[v]
                    Gi.add_edge(u, v)
        f.close()
        pickle.dump(Gi, open(selfpkl_file, 'wb'))
    return Gp, Gs, Gi, node_index

def load_graph_withtrack(data_dir):
    # loads nodes observed in any cascade.
    node_file = os.path.join(data_dir, 'seen_nodes.txt')
    with open(node_file, 'rb') as f:
        seen_nodes = [x.strip() for x in f]

    # builds node index
    node_index = {v: i for i, v in enumerate(seen_nodes)}
    node_reverse_index = {i: v for i, v in enumerate(seen_nodes)}

    # loads graph
    parent_file = os.path.join(data_dir, 'parentgraph.txt')
    sibling_file = os.path.join(data_dir, 'siblinggraph.txt')
    self_file = os.path.join(data_dir, 'selfgraph.txt')
    parentpkl_file = os.path.join(data_dir, 'parent.pkl')
    siblingpkl_file = os.path.join(data_dir, 'sibling.pkl')
    selfpkl_file = os.path.join(data_dir, 'self.pkl')

    if os.path.isfile(parentpkl_file):
        Gp = pickle.load(open(parentpkl_file, 'rb'))
        Gs = pickle.load(open(siblingpkl_file, 'rb'))
        Gi = pickle.load(open(selfpkl_file, 'rb'))
    else:
        Gp = nx.Graph()
        Gp.name = data_dir+'parent'
        n_nodes = len(node_index)
        Gp.add_nodes_from(range(n_nodes))
        with open(parent_file, 'rb') as f:
            f.next()
            for line in f:
                u, v = line.strip().split()
                if (u in node_index) and (v in node_index):
                    u = node_index[u]
                    v = node_index[v]
                    Gp.add_edge(u, v)
        f.close()
        pickle.dump(Gp, open(parentpkl_file, 'wb'))
        Gs = nx.Graph()
        Gs.name = data_dir + 'sibling'
        Gs.add_nodes_from(range(n_nodes))
        with open(sibling_file, 'rb') as f:
            f.next()
            for line in f:
                u, v = line.strip().split()
                if (u in node_index) and (v in node_index):
                    u = node_index[u]
                    v = node_index[v]
                    Gs.add_edge(u, v)
        f.close()
        pickle.dump(Gs, open(siblingpkl_file, 'wb'))

        Gi = nx.Graph()
        Gi.name = data_dir + 'self'
        Gi.add_nodes_from(range(n_nodes))
        with open(self_file, 'rb') as f:
            f.next()
            for line in f:
                u, v = line.strip().split()
                if (u in node_index) and (v in node_index):
                    u = node_index[u]
                    v = node_index[v]
                    Gi.add_edge(u, v)
        f.close()
        pickle.dump(Gi, open(selfpkl_file, 'wb'))
    return Gp, Gs, Gi, node_index, node_reverse_index

def convert_cascade_to_examples(sequence,
                                tssequence,
                                Gp=None,
                                Gs=None,
                                Gi=None,
                                inference=False):
    '''
        convert an event sequence to a set of training examples, {node 0}, {node 0, node 1}, ..., {node 0, ..., node N-2}, with label from node 1 to node N-1
        :param sequence: sequence is a sequence of event ids
        :param G: nx.Graph(), the graph constructed by relations
        :param inference: a boolean variable, if false, we stores the next event id as the label (the one to be predicted), if true (what?)
        :return: a list of dictionaries, each dictionary correspond to a training example, with sequence, topo_mask and label
    '''
    length = len(sequence)

    # grows the series of dags incrementally.
    examples = []
    dagp = nx.DiGraph()
    dags = nx.DiGraph()
    dagi = nx.DiGraph()
    dagt = nx.DiGraph()
    #maxtgr = 0
    for i, node in enumerate(sequence):
        # grows the DAG.
        prefix = sequence[: i + 1]
        #prefixtime = tssequence[: i + 1]
        dagp.add_node(node)
        dags.add_node(node)
        dagi.add_node(node)
        dagt.add_node(node)
        predecessors = set(Gp[node]) & set(prefix)
        dagp.add_edges_from(
            [(v, node) for v in predecessors])
        predecessorss = set(Gs[node]) & set(prefix)
        dags.add_edges_from(
            [(v, node) for v in predecessorss])
        predecessorsi = set(Gi[node]) & set(prefix)
        dagi.add_edges_from(
            [(v, node) for v in predecessorsi])

        # (optional) adds chronological edges
        if i > 0:
            dagt.add_edge(sequence[i - 1], node)

        if i == length - 1 and not inference:
            return examples

        if i < length - 1 and inference:
            continue

        # compiles example from DAG.
        node_pos = defaultdict(list)
        for k, va in [(v, i) for i, v in enumerate(prefix)]:
            node_pos[k].append(va)
        prefix_len = len(prefix)
        topop_mask = np.zeros((prefix_len, prefix_len), dtype=np.int)
        topos_mask = np.zeros((prefix_len, prefix_len), dtype=np.int)
        topoi_mask = np.zeros((prefix_len, prefix_len), dtype=np.int)
        topot_mask = np.zeros((prefix_len, prefix_len), dtype=np.int)

        for i_v, v in enumerate(prefix):
            i_p = []
            for x in dagp.predecessors(v):
                tmp = np.asarray(node_pos[x])
                if tmp[tmp < i_v].size > 0:
                    i_p.append(tmp[tmp < i_v].max())
            for sip in i_p:
                if float(tssequence[i_v]) - float(tssequence[sip]) > 0.1:
                    i_p.remove(sip)
            topop_mask[i_v, i_p] = 1
            i_p = []
            for x in dags.predecessors(v):
                tmp = np.asarray(node_pos[x])
                if tmp[tmp < i_v].size > 0:
                    i_p.append(tmp[tmp < i_v].max())
            for sip in i_p:
                if float(tssequence[i_v]) - float(tssequence[sip]) > 1:
                    i_p.remove(sip)
            topos_mask[i_v, i_p] = 1
            i_p = []
            for x in dagi.predecessors(v):
                tmp = np.asarray(node_pos[x])
                if tmp[tmp < i_v].size > 0:
                    i_p.append(tmp[tmp < i_v].max())
            #for sip in i_p:
            #    if float(tssequence[i_v]) - float(tssequence[sip]) > 1:
            #        i_p.remove(sip)
            topoi_mask[i_v, i_p] = 1
            i_p = []
            for x in dagt.predecessors(v):
                tmp = np.asarray(node_pos[x])
                if tmp[tmp < i_v].size >0:
                    i_p.append(tmp[tmp < i_v].max())
            topot_mask[i_v, i_p] = 1

        #calculate the timegap mask, to see what time gap range does this sequence fall into, if contains larger timegap, the predicted one will tend to bigger
        #tg_mask = np.zeros((1,4), dtype=np.int)
        tg_mask = 0
        '''
        if i==0:
            #maxtgr = 0
            tg_mask = 0
        else:
            ctgr = get_range((float(tssequence[i]) - float(tssequence[i-1])))
            #maxtgr = max(maxtgr, ctgr)
            tg_mask = ctgr
        '''
        if not inference:
            label = sequence[i + 1]
            tslabel = (float(tssequence[i + 1]) - float(tssequence[i]))/1.  #time gap of the next event to the current event

        else:
            label = None

        example = {'sequence': prefix,
                   'topop_mask': topop_mask,
                   'topos_mask': topos_mask,
                   'topoi_mask': topoi_mask,
                   'topot_mask': topot_mask,
                   'label': label,
                   #'tssequence': prefixtime,
                   'tslabel': tslabel,
                   'tg_mask':tg_mask}

        if not inference:
            examples.append(example)
        else:
            return example


def load_examples(data_dir,
                  dataset=None,
                  Gp=None,
                  Gs=None,
                  Gi=None,
                  node_index=None,
                  maxlen=None,
                  keep_ratio=1.):
    '''
    Load the train/dev/test data
    convert the sequences stored in files to a list of training examples, each L length sequence will be convert to L-1 examples
    Return: list of example tuples
    :param data_dir:
    :param dataset:
    :param G:
    :param node_index:
    :param maxlen: the maximum length a sequence is read
    :param keep_ratio: how many percetage of training examples to use
    :return:
    '''

    pkl_path = os.path.join(data_dir, dataset + '.pkl')
    if os.path.isfile(pkl_path):
        print 'pickle exists.'
        examples = pickle.load(open(pkl_path, 'rb'))
    else:
        # loads cascades
        filename = os.path.join(data_dir, dataset + '.txt')
        examples = []
        with codecs.open(filename, 'r', encoding='utf-8') as input_file:
            for line_index, line in enumerate(input_file):
                # parses the input line.
                allitems = line.strip().split(' ')
                sequence = allitems[::2]
                tssequence = allitems[1::2]
                if maxlen is not None:
                    sequence = sequence[:maxlen]
                    tssequence = tssequence[:maxlen]
                sequence = [node_index[x] for x in sequence]

                sub_examples = convert_cascade_to_examples(sequence, tssequence, Gp=Gp, Gs= Gs, Gi = Gi)
                examples.extend(sub_examples)

        pickle.dump(examples, open(pkl_path, 'wb'))

    n_samples = len(examples)
    indices = np.random.choice(n_samples, int(
        n_samples * keep_ratio), replace=False)
    sampled_examples = [examples[i] for i in indices]
    return sampled_examples

def load_examples_seq(data_dir,
                  dataset=None,
                  Gp=None,
                  Gs=None,
                  Gi=None,
                  node_index=None,
                  maxlen=None,
                  keep_ratio=1.):
    '''
    Load the train/dev/test data
    convert the sequences stored in files to a list of training examples, each L length sequence will be convert to L-1 examples
    Return: list of example tuples
    :param data_dir:
    :param dataset:
    :param G:
    :param node_index:
    :param maxlen: the maximum length a sequence is read
    :param keep_ratio: how many percetage of training examples to use
    :return:
    '''

    pkl_path = os.path.join(data_dir, dataset + '.pkl')
    if os.path.isfile(pkl_path):
        print 'pickle exists.'
        examples = pickle.load(open(pkl_path, 'rb'))
    else:
        # loads cascades
        filename = os.path.join(data_dir, dataset + '.txt')
        examples = []
        with codecs.open(filename, 'r', encoding='utf-8') as input_file:
            for line_index, line in enumerate(input_file):
                # parses the input line.
                allitems = line.strip().split(' ')
                sequence = allitems[::2]
                tssequence = allitems[1::2]
                if maxlen is not None:
                    sequence = sequence[:maxlen]
                    tssequence = tssequence[:maxlen]
                sequence = [node_index[x] for x in sequence]

                sub_examples = convert_cascade_to_examples(sequence, tssequence, Gp=Gp, Gs= Gs, Gi = Gi)
                examples.extend(sub_examples)
    return examples

def prepare_minibatch(tuples, inference=False, options=None):
    '''
    produces a mini-batch of data in format required by model.
    :param tuples: the tuples of dictionary (examples)
    :param inference:
    :param options:
    :return: the data in the format required by theano model
    '''
    seqs = [t['sequence'] for t in tuples]
    #tsseqs = [t['tssequence'] for t in tuples]
    lengths = map(len, seqs)
    n_timesteps = max(lengths)
    n_samples = len(tuples)

    # prepare sequences data
    seqs_matrix = np.zeros((n_timesteps, n_samples)).astype('int32')
    for i, seq in enumerate(seqs):
        seqs_matrix[: lengths[i], i] = seq

    '''
    # prepare timestamps data
    tsseqs_matrix = np.zeros((n_timesteps, n_samples)).astype('float32')
    for i, tsseq in enumerate(tsseqs):
        tsseqs_matrix[: lengths[i], i] = tsseq
    '''

    # prepare topo-masks data
    topop_masks = [t['topop_mask'] for t in tuples]
    topop_masks_tensor = np.zeros(
        (n_timesteps, n_samples, n_timesteps)).astype(config.floatX)
    for i, topop_mask in enumerate(topop_masks):
        topop_masks_tensor[: lengths[i], i, : lengths[i]] = topop_mask

    topos_masks = [t['topos_mask'] for t in tuples]
    topos_masks_tensor = np.zeros(
        (n_timesteps, n_samples, n_timesteps)).astype(config.floatX)
    for i, topos_mask in enumerate(topos_masks):
        topos_masks_tensor[: lengths[i], i, : lengths[i]] = topos_mask

    topoi_masks = [t['topoi_mask'] for t in tuples]
    topoi_masks_tensor = np.zeros(
        (n_timesteps, n_samples, n_timesteps)).astype(config.floatX)
    for i, topoi_mask in enumerate(topoi_masks):
        topoi_masks_tensor[: lengths[i], i, : lengths[i]] = topoi_mask

    topot_masks = [t['topot_mask'] for t in tuples]
    topot_masks_tensor = np.zeros(
        (n_timesteps, n_samples, n_timesteps)).astype(config.floatX)
    for i, topot_mask in enumerate(topot_masks):
        topot_masks_tensor[: lengths[i], i, : lengths[i]] = topot_mask

    # prepare sequence masks
    seq_masks_matrix = np.zeros((n_timesteps, n_samples)).astype(config.floatX)
    for i, length in enumerate(lengths):
        seq_masks_matrix[: length, i] = 1.

    # prepare time gap masks
    tg_masks = [t['tg_mask'] for t in tuples]
    tg_masks_matrix = np.zeros((n_samples, 1)).astype(config.floatX)
    for i, tg_mask in enumerate(tg_masks):
        tg_masks_matrix[i,tg_mask] = 1.

    # prepare labels data
    if not inference:
        labels = [t['label'] for t in tuples]
        labels_vector = np.array(labels).astype('int32')
        tslabels = [t['tslabel'] for t in tuples]
        tslabels_vector = np.array([tslabels]).astype('float32').T
    else:
        labels_vector = None

    return (seqs_matrix,
            seq_masks_matrix,
            topop_masks_tensor,
            topos_masks_tensor,
            topoi_masks_tensor,
            topot_masks_tensor,
            tg_masks_matrix,
            labels_vector,
            #tsseqs_matrix,
            tslabels_vector)


class Loader:
    def __init__(self, data, options=None):
        self.batch_size = options['batch_size']
        self.idx = 0
        self.data = data
        self.shuffle = options['shuffle_data']
        self.n = len(data)
        self.n_words = options['n_events']
        self.indices = np.arange(self.n, dtype="int32")
        self.options = options

    def __len__(self):
        return len(self.data) // self.batch_size + 1

    def __call__(self):
        if self.shuffle and self.idx == 0:
            np.random.shuffle(self.indices)

        batch_indices = self.indices[self.idx: self.idx + self.batch_size]
        batch_examples = [self.data[i] for i in batch_indices]

        self.idx += self.batch_size
        if self.idx >= self.n:
            self.idx = 0

        return prepare_minibatch(batch_examples,
                                 inference=False,
                                 options=self.options)
