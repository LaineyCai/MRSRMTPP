'''
Author: Hongyun Cai
Date: 2018.3.12
2018.3.14: add attention for all previous relational events
2018.3.15: instead of using scan to loop the timestep in a attention tensor, we use reshape tensor as a more efficient way to get rid of scan operation
2018.3.16: add time prediction loss, use AAAI method, gaussian penalty with the gt time and predicted time (predict use linear function)
2018.3.19: add time prediction for testing
2018.4.10: add multiple relation
Functionality: the structure GRU model
'''
from __future__ import print_function

import numpy as np
import theano
from theano import config
import theano.tensor as tensor
import math


# Set the random number generators' seeds for consistency
SEED = 123
np.random.seed(SEED)


def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def gru_layer(tparams, state_below, options, seq_masks=None, topop_masks=None, topos_masks=None, topoi_masks=None, topot_masks=None):
    '''
    The GRU model.
    state_below.shape (n_timesteps, n_samples, dim_proj)
    topo_masks.shape: (n_timesteps, n_samples, n_timesteps)

    Returns:
        a tensor of hidden states for all steps, has shape (n_timesteps, n_samples, dim_proj).
    '''
    n_timesteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert seq_masks is not None
    assert topop_masks is not None
    assert topos_masks is not None
    assert topoi_masks is not None
    assert topot_masks is not None

    def _step(index, seq_m_, topo_mp_, topo_ms_, topo_mi_, topo_mt_, x_, h_arr_):
        '''
        A GRU step.
        topo_m_.shape = (n_samples, n_timesteps)
        seq_m_.shape = (n_samples)
        x_.shape = (n_samples, dim_proj)
        h_arr_.shape shape = (n_timesteps, n_samples, dim_proj)
        '''
        # tranposes h_arr_ to have shape (n_samples, n_timesteps, dim_proj)
        # h_nb the list of hidden states of all relational precedors, has shape n_samples*n_timestep*dim_proj, for the n_timestamp where node is not relational, the value is zero
        h_nbp = (topo_mp_[:, :, None] * h_arr_.dimshuffle(1, 0, 2))
        #beta shape (n_sample, n_timesteps)
        betap = tensor.dot(tensor.tanh(tensor.dot(h_nbp.reshape([n_samples*n_timesteps, options['dim_proj']]), tparams['attp_q'])+tparams['attp_b']), tparams['attp_eta']).reshape([n_samples,n_timesteps])
        betap = tensor.exp(betap)
        alphap = betap / betap.sum(axis=-1, keepdims=True)

        h_nbi = (topo_mi_[:, :, None] * h_arr_.dimshuffle(1, 0, 2))
        betai = tensor.dot(tensor.tanh(
            tensor.dot(h_nbi.reshape([n_samples * n_timesteps, options['dim_proj']]), tparams['atti_q']) + tparams[
                'atti_b']), tparams['atti_eta']).reshape([n_samples, n_timesteps])
        betai = tensor.exp(betai)
        alphai = betai / betai.sum(axis=-1, keepdims=True)

        h_nbs = (topo_ms_[:, :, None] * h_arr_.dimshuffle(1, 0, 2))
        betas = tensor.dot(tensor.tanh(
            tensor.dot(h_nbs.reshape([n_samples * n_timesteps, options['dim_proj']]), tparams['atts_q']) + tparams[
                'atts_b']), tparams['atts_eta']).reshape([n_samples, n_timesteps])
        betas = tensor.exp(betas)
        alphas = betas / betas.sum(axis=-1, keepdims=True)

        h_nbt = (topo_mt_[:, :, None] * h_arr_.dimshuffle(1, 0, 2))


        # h_sum_ has shape n_samples * dim_proj
        h_sump = (alphap[:,:,None] * h_nbp).sum(axis=1)
        h_sums = (alphas[:, :, None] * h_nbs).sum(axis=1)
        h_sumi = (alphai[:, :, None] * h_nbi).sum(axis=1)
        h_sumt = h_nbt.sum(axis=1)

        # shape (n_samples, )
        beta_ap = tensor.dot(tensor.tanh(tensor.dot(h_sump, tparams['atta_q'])+tparams['atta_b']), tparams['atta_eta'])
        beta_as = tensor.dot(tensor.tanh(tensor.dot(h_sums, tparams['atta_q']) + tparams['atta_b']),
                             tparams['atta_eta'])
        beta_ai = tensor.dot(tensor.tanh(tensor.dot(h_sumi, tparams['atta_q']) + tparams['atta_b']),
                             tparams['atta_eta'])
        beta_at = tensor.dot(tensor.tanh(tensor.dot(h_sumt, tparams['atta_q']) + tparams['atta_b']),
                             tparams['atta_eta'])
        beta_ap = tensor.exp(beta_ap)
        beta_as = tensor.exp(beta_as)
        beta_ai = tensor.exp(beta_ai)
        beta_at = tensor.exp(beta_at)
        beta_sum = beta_ap + beta_as+ beta_ai + beta_at
        # shape (n_samples, )
        alpha_ap = beta_ap / beta_sum
        alpha_as = beta_as / beta_sum
        alpha_ai = beta_ai / beta_sum
        alpha_at = beta_at / beta_sum

        h_sum = alpha_ap[:,None] * h_sump + alpha_as[:, None] * h_sums + alpha_ai[:, None]*h_sumi + alpha_at[:, None]*h_sumt

        #h_sum = (topo_m_[:, :, None] * h_arr_.dimshuffle(1, 0, 2)).sum(axis=1)

        z = tensor.nnet.sigmoid(tensor.dot(x_, tparams['gru_Wz']) + tensor.dot(h_sum, tparams['gru_Uz']) + tparams['gru_bz'])
        r = tensor.nnet.sigmoid(
            tensor.dot(x_, tparams['gru_Wr']) + tensor.dot(h_sum, tparams['gru_Ur']) + tparams['gru_br'])
        h_ = tensor.tanh(tensor.dot(x_, tparams['gru_Wh']) + tensor.dot(h_sum*r, tparams['gru_Uh']) + tparams['gru_bh'])
        h = h_sum * (1. - z) + h_*z
        h = seq_m_[:, None] * h

        h_arr_ = tensor.set_subtensor(h_arr_[index, :], h)

        return h_arr_

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[tensor.arange(n_timesteps),
                                           seq_masks,
                                           topop_masks,
                                           topos_masks,
                                           topoi_masks,
                                           topot_masks,
                                           state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_timesteps, n_samples, dim_proj)],
                                name='gru_layers',
                                n_steps=n_timesteps)

    return rval[-1]


def build_model(tparams, timetparams, options):
    '''
    Builds Structure-GRU model.
    Returns a dictionary of outlet symbols
    '''
    # Set up input symbols with shapes:
    #   seqs.shape = (n_timesteps, n_samples), each example is a column vector
    #   seq_masks.shape = (n_timesteps, n_samples), each column is the mask for an example
    #   topo_masks.shape = (n_timesteps, n_samples, n_timesteps), the second dimension indicate the id of the topo mask (n_timesteps * n_timesteps)
    #   labels.shape = (n_samples,), the label for each example (the next event id)
    seqs = tensor.matrix('seqs', dtype='int32')
    #tsseqs = tensor.matrix('tsseqs', dtype='float32')
    seq_masks = tensor.matrix('seq_masks', dtype=config.floatX)
    topop_masks = tensor.tensor3('topop_masks', dtype=config.floatX)
    topos_masks = tensor.tensor3('topos_masks', dtype=config.floatX)
    topoi_masks = tensor.tensor3('topoi_masks', dtype=config.floatX)
    topot_masks = tensor.tensor3('topot_masks', dtype=config.floatX)
    tg_masks = tensor.matrix('tg_masks', dtype=config.floatX)
    labels = tensor.vector('labels', dtype='int32')
    #tslabels = tensor.vector('tslabels', dtype='float32')
    tslabels = tensor.matrix('tslabels', dtype='float32')

    inputs = [seqs, seq_masks, topop_masks, topos_masks, topoi_masks, topot_masks]

    n_timesteps = seqs.shape[0]
    n_samples = seqs.shape[1]

    # embedding lookup.
    embs = tparams['Eemb'][seqs.flatten()].reshape([n_timesteps,
                                                    n_samples,
                                                    options['dim_proj']])

    # run through gru layer.
    # h_arr.shape = (n_timesteps, n_samples, dim_proj)
    h_arr = gru_layer(tparams, embs, options, seq_masks=seq_masks, topop_masks=topop_masks, topos_masks=topos_masks, topoi_masks=topoi_masks, topot_masks=topot_masks)

    # mean pooling of hidden states, h_mean.shape=(n_samples, dim_proj)

    h_sum = (seq_masks[:, :, None] * h_arr).sum(axis=0)
    lengths = seq_masks.sum(axis=0)
    h_mean = h_sum / lengths[:, None]


    # decode h_mean into input to softmax
    s = tensor.dot(h_mean, tparams['W_ext']) + tparams['b_ext']

    probs = tensor.nnet.softmax(s)

    # predict time using h_mean, shape (n_samples,) + (1,)
    #shape (n_sample,1)
    tss = tensor.abs_(
        (tensor.dot(h_mean, timetparams['W_t']) * tg_masks).sum(axis=1, keepdims=True) + tensor.dot(tg_masks,
                                                                                                timetparams['b_t']))

    #tss = tensor.dot(h_mean, tparams['W_t'])
    #print(tensor.shape(tss).eval())
    #tss = tss + theano.tensor.tile(tparams['b_t'],(n_samples,))
    #tparams['b_t'] = tensor.addbroadcast(tparams['b_t'], 0)
    #tss = tss + tparams['b_t']

    # set up cost
    loss = tensor.nnet.nnet.categorical_crossentropy(probs, labels).mean()
    tsloss = tensor.mean((tss-tslabels)*(tss-tslabels))

    cost = loss
    timecost = tsloss

    # L2 penalty terms

    cost += options['weight_decay'] * (tparams['gru_Wr'] ** 2).sum()
    cost += options['weight_decay'] * (tparams['gru_Wh'] ** 2).sum()
    cost += options['weight_decay'] * (tparams['gru_Wz'] ** 2).sum()
    cost += options['weight_decay'] * (tparams['gru_Ur'] ** 2).sum()
    cost += options['weight_decay'] * (tparams['gru_Uh'] ** 2).sum()
    cost += options['weight_decay'] * (tparams['gru_Uz'] ** 2).sum()
    cost += options['weight_decay'] * (tparams['gru_br'] ** 2).sum()
    cost += options['weight_decay'] * (tparams['gru_bh'] ** 2).sum()
    cost += options['weight_decay'] * (tparams['gru_bz'] ** 2).sum()
    cost += options['weight_decay'] * (tparams['W_ext'] ** 2).sum()
    cost += options['weight_decay'] * (tparams['b_ext'] ** 2).sum()
    cost += options['weight_decay'] * (tparams['atta_q'] ** 2).sum()
    cost += options['weight_decay'] * (tparams['atta_b'] ** 2).sum()
    cost += options['weight_decay'] * (tparams['atta_eta'] ** 2).sum()

    timecost += options['weight_decay'] * (timetparams['W_t'] ** 2).sum()
    timecost += options['weight_decay'] * (timetparams['b_t'] ** 2).sum()


    # set up functions for inferencing
    f_prob = theano.function(inputs, probs, name='f_prob')
    f_pred = theano.function(inputs, probs.argmax(axis=1), name='f_pred')
    f_tprob = theano.function(inputs + [tg_masks], tss, name='f_tprob')

    return {'inputs': inputs,
            'labels': labels,
            'tslabels': tslabels,
            'tg_masks': tg_masks,
            'cost': cost,
            'timecost': timecost,
            'f_prob': f_prob,
            'f_pred': f_pred,
            'f_tprob': f_tprob,
            'data': inputs + [labels],
            'timedata': inputs + [tg_masks] + [tslabels]}
