from backend.utils import *
import numpy as np

__all__ = ['TlayersNN']


def TlayersNN(Players, Teff, g, met=0, opacity='rosseland'):
    '''

    :param Players: Pressure at different layers in dyne/cm^2, preferably logarithmically increasing
    :param Teff: Effective temperature in K
    :param g: Surface gravity in cm/s^2
    :param met: Metallicity
    :param opacity: average opacity - 'rosseland' or 'planck'
    :return: Temperature at different layers in K
    '''
    unpack = False
    if isinstance(Teff, (int, float)): Teff, unpack = [Teff], True
    if np.ndim(Players) == 1: Players = Players * np.ones((len(Teff), len(Players)))
    T = []
    for p, teff in zip(Players, Teff):
        tau = 1e-8
        t = [TtauNN(tau, teff)]
        tau = kappaNN(t, p[0], met, kappa=opacity) * p[0] / g
        for i in range(len(p) - 1):
            tau += kappaNN(t[i], p[i], met, kappa=opacity) * (p[i + 1] - p[i]) / g
            t = np.append(t, TtauNN(tau, teff))
        T.append(t[np.searchsorted(p, p)])
    return np.array(T)[0] if unpack else np.array(T)
