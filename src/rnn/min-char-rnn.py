# -*- coding: utf-8 -*-

"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License

Modified by Zhang Chen (@chansonzhang) to use python version 3.x and do some code polish
"""
import numpy as np

# data I/O
data = open('input.txt', 'r', encoding="utf-8").read()  # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# hyper-parameters
hidden_size = 100  # size of hidden layer of neurons
seq_length = 25  # number of steps to unroll the RNN for
learning_rate = 1e-1

assert len(data) > 10 * seq_length, "not enough data"

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
bh = np.zeros((hidden_size, 1))  # hidden bias
by = np.zeros((vocab_size, 1))  # output bias


def loss_fun(x, y, hidden):
    """
    x,y are both list of integers.
    hidden is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hidden)
    f_loss = 0  # loss
    # forward pass
    for t in range(len(x)):
        xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
        xs[t][x[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)  # hidden state
        ys[t] = np.dot(Why, hs[t]) + by  # unnormalized log probabilities for next chars
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilities for next chars
        f_loss += -np.log(ps[t][y[t], 0])  # softmax (cross-entropy loss)
    # backward pass: compute gradients going backwards
    d_w_xh, d_w_hh, d_w_hy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    d_bh, d_by = np.zeros_like(bh), np.zeros_like(by)
    d_h_next = np.zeros_like(hs[0])
    for t in reversed(range(len(x))):
        dy = np.copy(ps[t])
        dy[y[t]] -= 1  # back prop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
        d_w_hy += np.dot(dy, hs[t].T)
        d_by += dy
        dh = np.dot(Why.T, dy) + d_h_next  # back prop into h
        d_h_raw = (1 - hs[t] * hs[t]) * dh  # back prop through tanh non-linearity
        d_bh += d_h_raw
        d_w_xh += np.dot(d_h_raw, xs[t].T)
        d_w_hh += np.dot(d_h_raw, hs[t - 1].T)
        d_h_next = np.dot(Whh.T, d_h_raw)
    for d in [d_w_xh, d_w_hh, d_w_hy, d_bh, d_by]:
        np.clip(d, -5, 5, out=d)  # clip to mitigate exploding gradients
    return f_loss, d_w_xh, d_w_hh, d_w_hy, d_bh, d_by, hs[len(x) - 1]


def sample(h, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    idx_list = []
    for t in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        idx_list.append(ix)
    return idx_list


iteration, pos = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad
smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0
h_prev = np.zeros((hidden_size, 1))

while True:
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if pos + seq_length + 1 >= len(data):
        h_prev = np.zeros((hidden_size, 1))  # reset RNN memory
        pos = 0  # go from start of data
    data_inputs = [char_to_ix[ch] for ch in data[pos:pos + seq_length]]
    targets = [char_to_ix[ch] for ch in data[pos + 1:pos + seq_length + 1]]

    # sample from the model now and then
    if iteration % 100 == 0:
        sample_ix = sample(h_prev, data_inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt,))

    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, h_prev = loss_fun(data_inputs, targets, h_prev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if iteration % 100 == 0:
        print('iter %d, loss: %f' % (iteration, smooth_loss))  # print progress

    # perform parameter update with Adagrad
    for param, d_param, mem in zip([Wxh, Whh, Why, bh, by],
                                   [dWxh, dWhh, dWhy, dbh, dby],
                                   [mWxh, mWhh, mWhy, mbh, mby]):
        mem += d_param * d_param
        param += -learning_rate * d_param / np.sqrt(mem + 1e-8)  # adagrad update

    pos += seq_length  # move data pointer
    iteration += 1  # iteration counter
