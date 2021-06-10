import numpy as np
import scipy.special as special


class Sphere:
    """Class for predicting sphere scattering.
    """

    def __init__(self, material_type='elastic', radius=0.0254, material_params=None):
        MATERIAL_TYPE = {
            'elastic': self._form_function_elastic,
            'rigid_fix': self._form_function_rigidfix,
            'fluid': self._form_function_fluid,
            'shell': self._form_function_shell,
        }
        self.radius = radius  # sphere radius [m]
        self.material_type = material_type
        self.material_params = material_params  # sound speed and density contrast
        self._fm_func = MATERIAL_TYPE[self.material_type]  # function to calculate form function
        self.ka = None
        self.theta = None
        self.mode_num_max = None
        self._fm = None
        self._TS = None

    @property
    def fm(self):
        return self._fm

    @fm.setter
    def fm(self, val):
        self._fm = val

    @property
    def TS(self):
        if self._TS is None:
            print('TS or form function has not been calculated yet.')
        else:
            return self._TS

    @TS.setter
    def TS(self, val):
        self._TS = val

    @staticmethod
    def _Pn(n, x):
        """
        Legendre Polynomial: Pn(cos(theta)) = Pn(n,x).

        Parameters
        ----------
        n: order
        x: cos(theta)

        Returns
        -------
        Legendre Polynomial resulting from Pn(cos(theta)).
        """
        if np.abs(x).max() > 1:
            print('|x| must be smaller than 1')
            return -1

        pn = np.empty((x.size, n.size))
        pn[:, 0] = np.ones(x.size)
        pn[:, 1] = x
        for nn in np.arange(1, n.max()):
            pn[:, nn + 1] = ((2 * nn + 1) * x * pn[:, nn] - nn * pn[:, nn - 1]) / (nn + 1)

        return pn

    def _form_function_elastic(self):
        """
        Complex form function for elastic sphere as a function of scattering angle and ka.
        """
        n = np.arange(self.mode_num_max)
        pn = self._Pn(n, np.cos(self.theta / 180 * np.pi))
        nl = 2 * n + 1

        ka1 = self.ka
        ka2L = ka1 / self.material_params['hc']
        ka2s = ka1 / self.material_params['hs']

        jn1 = special.spherical_jn(np.expand_dims(n, 0), np.expand_dims(ka1, 1))
        yn1 = special.spherical_yn(np.expand_dims(n, 0), np.expand_dims(ka1, 1))
        djn1 = special.spherical_jn(np.expand_dims(n, 0), np.expand_dims(ka1, 1), derivative=True)
        dyn1 = special.spherical_yn(np.expand_dims(n, 0), np.expand_dims(ka1, 1), derivative=True)

        jn2L = special.spherical_jn(np.expand_dims(n, 0), np.expand_dims(ka2L, 1))
        djn2L = special.spherical_jn(np.expand_dims(n, 0), np.expand_dims(ka2L, 1), derivative=True)

        jn2s = special.spherical_jn(np.expand_dims(n, 0), np.expand_dims(ka2s, 1))
        djn2s = special.spherical_jn(np.expand_dims(n, 0), np.expand_dims(ka2s, 1), derivative=True)

        nn = n * n + n

        tan1 = -np.expand_dims(ka2L, 1) * djn2L / jn2L
        tan2 = -np.expand_dims(ka2s, 1) * djn2s / jn2s
        tan3 = -np.expand_dims(ka1, 1) * djn1 / jn1

        tan_beta = -np.expand_dims(ka1, 1) * dyn1 / yn1
        tan_del = -jn1 / yn1
        d1 = tan1 + 1
        d2 = nn - 1 - np.expand_dims(ka2s, 1) ** 2 / 2 + tan2

        term1a = tan1 / d1
        term1b = nn / d2
        term2a = (nn - np.expand_dims(ka2s, 1) ** 2 / 2 + 2 * tan1) / d1
        term2b = nn * (tan2 + 1) / d2

        td = -0.5 * np.expand_dims(ka2s, 1) ** 2 * (term1a - term1b) / (term2a - term2b)
        tan_phi = -td / self.material_params['g']
        tan_eta = tan_del * (tan_phi + tan3) / (tan_phi + tan_beta)
        cos_eta = 1 / np.sqrt(1 + tan_eta * tan_eta)
        sin_eta = tan_eta * cos_eta

        bn = sin_eta * (1j * cos_eta - sin_eta)
        s = nl * np.expand_dims(pn, 1) * np.expand_dims(bn, 0)
        y = s.sum(axis=2)

        return -1j * 2 * y / self.ka

    def _form_function_rigidfix(self):
        """
        Complex form function for rigid and fixed sphere as a function of scattering angle and ka.
        """

    def _form_function_fluid(self):
        """
        Complex form function for fluid sphere as a function of scattering angle and ka.
        """
        ka1 = self.ka
        ka2 = ka1 / self.material_params['h']
        n = np.arange(self.mode_num_max)
        pn1 = self._Pn(n, np.cos(self.theta / 180 * np.pi))
        nl = 2 * n + 1

        jn1 = special.spherical_jn(np.expand_dims(n, 0), np.expand_dims(ka1, 1))
        yn1 = special.spherical_yn(np.expand_dims(n, 0), np.expand_dims(ka1, 1))
        djn1 = special.spherical_jn(np.expand_dims(n, 0), np.expand_dims(ka1, 1), derivative=True)
        dyn1 = special.spherical_yn(np.expand_dims(n, 0), np.expand_dims(ka1, 1), derivative=True)

        jn2 = special.spherical_jn(np.expand_dims(n, 0), np.expand_dims(ka2, 1))
        djn2 = special.spherical_jn(np.expand_dims(n, 0), np.expand_dims(ka2, 1), derivative=True)

        term1 = djn2 * yn1 / (jn2 * djn1) - self.material_params['g'] * self.material_params['h'] * dyn1 / djn1
        term2 = djn2 * jn1 / (jn2 * djn1) - self.material_params['g'] * self.material_params['h']
        cn = term1 / term2
        bn = -1 / (1 + 1j * cn)

        s = nl * np.expand_dims(pn1, 1) * np.expand_dims(bn, 0)
        y = s.sum(axis=2)

        return -1j * 2 * y / self.ka

    def _form_function_shell(self):
        """
        Complex form function for spherical shell as a function of scattering angle and ka.
        """

    def get_TS(self, theta=None, ka=None, mode_num_max=None):
        """
        Calculate target strength (TS).

        Parameters
        ----------
        theta
            scattering angle [deg]: 0- forward scattering, 180-backscattering
            Default to backscattering direction.
        ka
            ka in medium.
        mode_num_max
            maximum number of modes to be calculated.
            Default to None and the calculation will use ``round(max(ka1))+10``.
        """
        # If no input argument is given, use previously computed form function
        if (ka is None) and (theta is None) and (mode_num_max is None):
            if self._fm is None:
                raise ValueError('Need to compute the scattering function at least once '
                                 'by calling ._get_TS(theta, ka, ...).')
            else:   # use form function computed previously
                self.TS = 20 * np.log10(np.abs(self._fm) / 2 * self.radius)

        # If one of the input arguments is given, recalculate self.fm
        else:
            if ka is not None:
                self.ka = ka
            else:
                self.ka = np.linspace(0, 30, 1000)
                print('Use default ka=np.linspace(0, 30, 1000).')

            if theta is not None:
                self.theta = theta
            else:
                self.theta = 180
                print('Use default theta=180 deg -- backscattering direction.')

            if mode_num_max is None:
                self.mode_num_max = np.ceil(self.ka.max()).astype('int') + 10
            else:
                self.mode_num_max = mode_num_max

            # Recalculate form function
            self.fm = self._fm_func()

            # Assign TS
            self.TS = 20 * np.log10(np.abs(self.fm) / 2 * self.radius)

        return self.TS
