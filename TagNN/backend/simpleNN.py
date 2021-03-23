import numpy as np, os
from sklearn.preprocessing import MinMaxScaler as mms


## Activation Functions ##
def sigmoid(x): return 1 / (1 + np.exp(-x))


def tansig(x): return 2. / (1 + np.exp(-2 * x)) - 1


def purelin(x): return x


## Derivative of Activation Functions ##
def derivatives_sigmoid(x): return x * (1 - x)


def derivatives_tansig(x): return 1 - x ** 2


def derivatives_purelin(x): return np.zeros(np.shape(x))


class NN_1hid:
    def __init__(self, net):
        self.net = net
        self.wh = net['wh']
        self.bh = net['bh']
        self.wout = net['wout']
        self.bout = net['bout']
        self.act = net['activation']
        self.ipscaler = net['scaler'][0]
        self.opscaler = net['scaler'][1]
        self.X = None
        self.pred = None

    def forward(self, X):
        self.X = self.ipscaler.transform(np.reshape(X, (-1, self.wh.shape[0])))
        hidden_layer_input = np.dot(self.X, self.wh) + self.bh
        actfunc = {'sigmoid': sigmoid, 'tansig': tansig, 'purelin': purelin}
        if type(self.act) == str:
            self.actfunch, self.actfunco = self.act, self.act
        elif isinstance(self.act, (list, tuple)):
            self.actfunch, self.actfunco = self.act[0], self.act[-1]
        self.hiddenlayer_activations = actfunc[self.actfunch](hidden_layer_input)
        output_layer_input = np.dot(self.hiddenlayer_activations, self.wout) + self.bout
        self.pred = self.opscaler.inverse_transform(actfunc[self.actfunco](output_layer_input))
        return self.pred

    def backward(self, Y, lr, ):
        if self.pred is None: raise RuntimeError(
            'backward process can\'t run without the forward process run prior.')
        E = self.opscaler.transform(np.reshape(Y, self.pred.shape)) - self.opscaler.transform(self.pred)
        derivfunc = {'sigmoid': derivatives_sigmoid, 'tansig': derivatives_tansig, 'purelin': derivatives_purelin}
        slope_output_layer = derivfunc[self.actfunco](self.pred)
        slope_hidden_layer = derivfunc[self.actfunch](self.hiddenlayer_activations)
        d_output = E * slope_output_layer
        Error_at_hidden_layer = d_output.dot(self.wout.T)
        d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
        self.wout += self.hiddenlayer_activations.T.dot(d_output) * lr
        self.bout += np.sum(d_output, axis=0, keepdims=True) * lr
        self.wh += self.X.T.dot(d_hiddenlayer) * lr
        self.bh += np.sum(d_hiddenlayer, axis=0, keepdims=True) * lr
        self.net['wh'], self.net['bh'], self.net['wout'], self.net['bout'] = self.wh, self.bh, self.wout, self.bout


def single_hidden_layer_train(X, Y, epoch=150, lr=0.1, hiddenlayer_neurons=3, activation='sigmoid'):
    X, Y = np.reshape(X, (len(X), -1)), np.reshape(Y, (len(Y), -1))
    inputlayer_neurons = X.shape[1]  # number of features in data set
    output_neurons = Y.shape[1]
    if X.shape[0] != Y.shape[0]: raise IOError(
        'The number of input samples ({}) is not equal to output samples ({})'.format(X.shape[0], Y.shape[0]))
    wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
    bh = np.random.uniform(size=(1, hiddenlayer_neurons))
    wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
    bout = np.random.uniform(size=(1, output_neurons))
    ipscaler = mms(feature_range=(-1, 1))
    ipscaler.fit(X)
    opscaler = mms(feature_range=(-1, 1))
    opscaler.fit(Y)
    net = dict(wh=wh, bh=bh, wout=wout, bout=bout, activation=activation, scaler=(ipscaler, opscaler))
    NNtask = NN_1hid(net)

    for i in range(epoch):
        NNtask.forward(X)
        NNtask.backward(Y, lr)
    NNtask.forward(X)
    return NN_1hid


def trainNtest(Xtrain, Ytrain, Xtest, epoch=150, lr=0.1, hiddenlayer_neurons=20, activation='sigmoid'):
    if Xtrain == [] or Ytrain == []: return [None] * 3
    funcobj = single_hidden_layer_train(Xtrain, Ytrain, epoch, lr, hiddenlayer_neurons, activation)
    trainpred = funcobj.pred
    if Xtest != []:
        testpred = funcobj.forward(Xtest)
    return funcobj, trainpred, testpred
