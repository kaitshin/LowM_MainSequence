from datetime import date

import numpy as np
import matplotlib.pyplot as plt

from ..NB_errors import mag_combine
from ..NB_errors import filt_ref, dNB, lambdac, dBB, epsilon

filters = ['NB704', 'NB711', 'NB816', 'NB921', 'NB973']
cont0 = [r'$R_Ci^{\prime}$', r'$R_Ci^{\prime}$', r'$i^{\prime}z^{\prime}$',
         r'$z^{\prime}$', r'$z^{\prime}$']
NB_filt = np.array([xx for xx in range(len(filt_ref)) if 'NB' in filt_ref[xx]])
for arr in ['filt_ref', 'dNB', 'lambdac', 'dBB', 'epsilon']:
    cmd1 = arr + ' = np.array('+arr+')'
    exec(cmd1)
    cmd2 = arr + ' = '+arr+'[NB_filt]'
    exec(cmd2)

import logging
formatter = logging.Formatter('%(asctime)s - %(module)12s.%(funcName)20s - %(levelname)s: %(message)s')

# Colors for each separate points on avg_sigma plots
avg_sig_ctype = ['m', 'r', 'g', 'b', 'k']

# Common text for labels
EW_lab = r'$\log({\rm EW}/\AA)$'
Flux_lab = r'$\log(F_{{\rm H}\alpha})$'
M_lab = r'$\log(M_{\star}/M_{\odot})$'
SFR_lab = r'$\log({\rm SFR}[{\rm H}\alpha]/M_{\odot}\,{\rm yr}^{-1})$'

EW_bins = np.arange(0.2, 3.0, 0.2)
Flux_bins = np.arange(-17.75, -14.00, 0.25)
sSFR_bins = np.arange(-11.0, -6.0, 0.2)
SFR_bins = np.arange(-5.0, 2.0, 0.2)
# Colors for each separate points on avg_sigma plots

cmap_sel = plt.cm.Blues
cmap_nosel = plt.cm.Reds

# Limiting magnitudes for NB and BB data
m_NB = np.array([26.7134 - 0.047, 26.0684, 26.9016 + 0.057, 26.7088 - 0.109, 25.6917 - 0.051])
m_BB1 = np.array([28.0829, 28.0829, 27.7568, 26.8250, 26.8250])
m_BB2 = np.array([27.7568, 27.7568, 26.8250, 00.0000, 00.0000])
cont_lim = mag_combine(m_BB1, m_BB2, epsilon)

# Minimum NB excess color for selection
minthres = [0.15, 0.15, 0.15, 0.2, 0.25]


class MLog:
    """
    Main class to log information to stdout and ASCII file

    To execute:
    mylog = MLog(dir0)._get_logger()

    Parameters
    ----------
    dir0 : str
      Full path for where log files should be placed

    Returns
    -------

    Notes
    -----
    Created by Chun Ly, 2 October 2019
    """

    def __init__(self, dir0, str_date):
        self.LOG_FILENAME = dir0 + 'completeness_analysis.' + str_date + '.log'
        self._log = self._get_logger()

    def _get_logger(self):
        loglevel = logging.INFO
        log = logging.getLogger(self.LOG_FILENAME)
        if not getattr(log, 'handler_set', None):
            log.setLevel(logging.INFO)
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            log.addHandler(sh)

            fh = logging.FileHandler(self.LOG_FILENAME)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            log.addHandler(fh)

            log.setLevel(loglevel)
            log.handler_set = True
        return log


def get_date(debug=False):
    """Return suffix with date stamp (month and date)"""

    today0 = date.today()
    str_date = "%02i%02i" % (today0.month, today0.day)
    if debug:
        str_date += ".debug"

    return str_date