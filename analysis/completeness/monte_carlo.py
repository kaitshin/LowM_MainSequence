import numpy as np


def mock_ones(arr0, Nmock):
    """
    Generate (Nmock,Ngal) array using np.ones() to repeat
    """

    return np.ones((Nmock, 1)) * arr0


def random_mags(t_seed, rand_shape, mag_ref, sig_ref):
    """
    Generate randomized array of magnitudes based on ref values and sigma
    """

    N_rep = rand_shape[0]

    np.random.seed(t_seed)
    return mock_ones(mag_ref, N_rep) + np.random.normal(size=rand_shape) * \
           mock_ones(sig_ref, N_rep)
