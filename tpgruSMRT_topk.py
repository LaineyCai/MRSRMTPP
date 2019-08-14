'''
Author: Hongyun Cai
Date: 2018.03.13
2018.03.14: add attention
2018.03.19: add time prediction for testing
'''
import numpy as np
import networkx as nx
import theano
from theano import config
from collections import OrderedDict
import timeit
import six.moves.cPickle as pickle
import downhill
import metrics
import pprint

import data_utilsSMRT
import tpgruSMRT_model


def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)

def init_params_weight(row,column):
    """
    initialize the parameters of a matrix where row may not equal to column
    """
    W = np.random.rand(row, column)
    W = W*2.0-1.0
    return W.astype(config.floatX)  # @UndefinedVariable


def init_params(options):
    """
    Initializes values of shared variables.
    """
    params = OrderedDict()

    # event embedding, shape = (n_events, dim_proj)
    randn = np.random.randn(options['n_events'],
                            options['dim_proj'])
    params['Eemb'] = (0.1 * randn).astype(config.floatX)

    # shape = dim_proj * dim_proj
    gru_Wz = ortho_weight(options['dim_proj'])
    params['gru_Wz'] = gru_Wz
    gru_Wh = ortho_weight(options['dim_proj'])
    params['gru_Wh'] = gru_Wh
    gru_Wr = ortho_weight(options['dim_proj'])
    params['gru_Wr'] = gru_Wr

    # shape = dim_proj * dim_proj
    gru_Uz = ortho_weight(options['dim_proj'])
    params['gru_Uz'] = gru_Uz
    gru_Uh = ortho_weight(options['dim_proj'])
    params['gru_Uh'] = gru_Uh
    gru_Ur = ortho_weight(options['dim_proj'])
    params['gru_Ur'] = gru_Ur

    # shape = dim_proj
    gru_bz = np.random.rand(options['dim_proj']).astype(config.floatX)-0.5
    params['gru_bz'] = gru_bz
    gru_bh = np.random.rand(options['dim_proj']).astype(config.floatX)-0.5
    params['gru_bh'] = gru_bh
    gru_br = np.random.rand(options['dim_proj']).astype(config.floatX)-0.5
    params['gru_br'] = gru_br

    # for attention
    attp_q = init_params_weight(options['dim_proj'], options['dim_att'])
    params['attp_q'] = attp_q
    attp_b = np.random.rand(options['dim_att'], ).astype(config.floatX) - 0.5
    params['attp_b'] = attp_b
    attp_eta = np.random.rand(options['dim_att'], ).astype(config.floatX) - 0.5
    params['attp_eta'] = attp_eta

    atts_q = init_params_weight(options['dim_proj'], options['dim_att'])
    params['atts_q'] = atts_q
    atts_b = np.random.rand(options['dim_att'], ).astype(config.floatX) - 0.5
    params['atts_b'] = atts_b
    atts_eta = np.random.rand(options['dim_att'], ).astype(config.floatX) - 0.5
    params['atts_eta'] = atts_eta

    atti_q = init_params_weight(options['dim_proj'], options['dim_att'])
    params['atti_q'] = atti_q
    atti_b = np.random.rand(options['dim_att'], ).astype(config.floatX) - 0.5
    params['atti_b'] = atti_b
    atti_eta = np.random.rand(options['dim_att'], ).astype(config.floatX) - 0.5
    params['atti_eta'] = atti_eta

    atta_q = init_params_weight(options['dim_proj'], options['dim_att'])
    params['atta_q'] = atta_q
    atta_b = np.random.rand(options['dim_att'], ).astype(config.floatX) - 0.5
    params['atta_b'] = atta_b
    atta_eta = np.random.rand(options['dim_att'], ).astype(config.floatX) - 0.5
    params['atta_eta'] = atta_eta

    # decoding matrix for external influences
    W_ext = init_params_weight(options['dim_proj'],
                               options['n_events'])
    params['W_ext'] = W_ext
    dec_b = np.random.rand(options['n_events']).astype(config.floatX)-0.5
    params['b_ext'] = dec_b.astype(config.floatX)

    return params

def init_timeparams(options):
    """
    Initializes values of shared variables.
    """
    params = OrderedDict()
    # for time prediction
    '''
    W_t = np.zeros(options['dim_proj'])
    params['W_t'] = W_t.astype(config.floatX)
    b_t = np.zeros(1)
    params['b_t'] = b_t.astype(config.floatX)
    '''
    W_t = init_params_weight(options['dim_proj'], 1)
    params['W_t'] = W_t.astype(config.floatX)
    b_t = init_params_weight(1, 1)
    params['b_t'] = b_t.astype(config.floatX)
    # w_g = np.zeros(1)
    # params['w_g'] = w_g.astype(config.floatX)

    return params


def init_tparams(params):
    '''
    Set up Theano shared variables.
    '''
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def evaluate_topk(f_prob, test_loader, f_tprob, tdim, node_reverse_index, savefile, k_list=[1,2,3,4,5,6,7,8,9,10]):
    '''
    Evaluates trained model.
    '''
    n_batches = len(test_loader)
    y = None
    y_prob = None
    yt = None
    yt_prob = None
    alltopk = None

    for _ in range(n_batches):
        batch_data = test_loader()
        y_ = batch_data[-2]
        y_prob_ = f_prob(*batch_data[:-3])
        yt_ = batch_data[-1]
        yt_prob_ = f_tprob(*batch_data[:-2])

        topk = np.zeros((y_.size, 10), dtype=int)

        # excludes activated nodes when predicting.
        for i, p in enumerate(y_prob_):
            length = int(np.sum(batch_data[1][:, i]))
            #sequence = batch_data[0][: length, i]
            #assert y_[i] not in sequence, str(sequence) + str(y_[i])
            #p[sequence] = 0.
            y_prob_[i, :] = p / float(np.sum(p))
            tmp_prob = p / float(np.sum(p))
            topk_ = sorted(range(tmp_prob.size), key=lambda k: tmp_prob[k], reverse=True)[:10]
            topk[i, :] = [node_reverse_index[d] for d in topk_]

        if y_prob is None:
            y_prob = y_prob_
            y = y_
            yt_prob = yt_prob_
            yt = yt_
            alltopk = topk
        else:
            y = np.concatenate((y, y_), axis=0)
            y_prob = np.concatenate((y_prob, y_prob_), axis=0)
            yt = np.concatenate((yt, yt_), axis=0)
            yt_prob = np.concatenate((yt_prob, yt_prob_), axis=0)
            alltopk = np.concatenate((alltopk, topk), axis=0)

    np.savetxt(savefile + "topk", alltopk, fmt='%d', delimiter=',')

    return metrics.portfolio(y_prob, y, yt_prob, yt, tdim, k_list=k_list)


def test_gettopk(data_dir='data/smrt/',
          dim_proj=64,
          dim_att=32,
          maxlen=None,
          batch_size=700,
          keep_ratio=1.,
          shuffle_data=False,
          learning_rate=0.0005,
          global_steps=50000,
          disp_freq=100,
          save_freq=300,
          test_freq=300,
          saveto_file='params.npz',
          weight_decay=0.0005,
          sigmasqr = 1,
          tdim = 1.,
          reload_model=False,
          train=True):
    """
    MRSRMTPP model loading.
    tdim: scale time down by how many times
    """
    options = locals().copy()
    saveto = data_dir + saveto_file
    tmsaveto = data_dir + 'timeparams.npz'

    # loads graph
    Gp, Gs, Gi, node_index, node_reverse_index = data_utilsSMRT.load_graph_withtrack(data_dir)
    options['n_events'] = len(node_index)

    print options

    # creates and initializes shared variables.
    print 'Initializing variables...'
    params = init_params(options)
    print 'reusing saved model.'
    load_params(saveto, params)
    tparams = init_tparams(params)

    timeparams = init_timeparams(options)
    print 'reusing saved model.'
    load_params(tmsaveto, timeparams)
    timetparams = init_tparams(timeparams)

    # builds Topo-LSTM model
    print 'Building model...'
    model = tpgruSMRT_model.build_model(tparams, timetparams, options)

    print 'Loading test data...'
    test_examples = data_utilsSMRT.load_examples_seq(data_dir,
                                             dataset='test',
                                             node_index=node_index,
                                             maxlen=maxlen,
                                             Gp=Gp,
                                             Gs=Gs,
                                             Gi=Gi)
    test_loader = data_utilsSMRT.Loader(test_examples, options=options)
    print 'Loaded %d test examples' % len(test_examples)

    scores = evaluate_topk(model['f_prob'], test_loader, model['f_tprob'], options['tdim'], node_reverse_index, data_dir)
    print 'eval scores: ', scores
    pprint.pprint(scores)


if __name__ == '__main__':
    test_gettopk(data_dir='data/smrt/', dim_proj=64, keep_ratio=1.)
