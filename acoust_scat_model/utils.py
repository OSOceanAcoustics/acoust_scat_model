import numpy as np


def Pn(n, x):
    """Legendre Polynomial: Pn(cos(theta)) = Pn(n,x)
    """
    if np.abs(x).max() > 1:
        print('|x| must be smaller than 1')
        return -1
    pn = np.empty((x.size, n.size))
    pn[:, 0] = np.ones(x.size)
    pn[:, 1] = x
    for nn in np.arange(1, n.max()):
        pn[:, nn+1] = ((2*nn+1) * x * pn[:, nn] - nn * pn[:, nn-1]) / (nn+1)
    return pn
