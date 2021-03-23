import numpy as np
import pickle as pkl
from . import simpleNN as snn
import os

rootdir = os.path.dirname(__file__)

__all__ = ['kappaNN', 'TtauNN']


def kappaNN(T, P, met, kappa='rosseland', mesh=False):
    if kappa.lower() == 'rosseland'[:len(kappa)]:
        filename = os.path.join(rootdir, 'datanet', 'KRnet.pk')
    elif kappa.lower() == 'planck'[:len(kappa)]:
        filename = os.path.join(rootdir, 'datanet', 'KPnet.pk')
    if mesh:
        P, T, met = np.meshgrid(P, T, met)
        ip = np.column_stack((np.log10(T).ravel(), np.log10(P).ravel(), met.ravel()))
        newshape = np.ones(T.ravel().shape)
    else:
        newshape = np.ones(np.shape(T)) * np.ones(np.shape(P)) * np.ones(np.shape(met))
        ip = np.column_stack(
            ((np.log10(T) * newshape).ravel(), (np.log10(P) * newshape).ravel(), (met * newshape).ravel()))
    net = pkl.load(open(filename, 'rb'))
    NNpro = snn.NN_1hid(net)
    NNpro.forward(ip)
    return 10 ** NNpro.pred.reshape(newshape.shape)


def TtauNN(tau, Teff, mesh=False):
    filename = os.path.join(rootdir, 'datanet', 'tauTnet.pk')
    if mesh:
        tau, Teff = np.meshgrid(tau, Teff)
        ip = np.column_stack((np.log(Teff).ravel(), np.log(tau).ravel()))
        newshape = np.ones(tau.ravel().shape)
    else:
        newshape = np.ones(np.shape(tau)) * np.ones(np.shape(Teff))
        ip = np.column_stack(((np.log(Teff) * newshape).ravel(), (np.log(tau) * newshape).ravel()))
    net = pkl.load(open(filename, 'rb'))
    NNpro = snn.NN_1hid(net)
    NNpro.forward(ip)
    return np.exp(NNpro.pred.reshape(newshape.shape)) * np.ravel(Teff)
