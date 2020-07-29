from datetime import date

import numpy as np
import matplotlib.pyplot as plt

from ..NB_errors import filt_ref, dNB, lambdac, dBB, epsilon

import logging
formatter = logging.Formatter('%(asctime)s - %(module)12s.%(funcName)20s - %(levelname)s: %(message)s')

# Strips out IA filters
NB_filt = np.array([xx for xx in range(len(filt_ref)) if 'NB' in filt_ref[xx]])
filter_vars = [filt_ref, dNB, lambdac, dBB, epsilon]
filter_vars_name = ['filt_ref', 'dNB', 'lambdac', 'dBB', 'epsilon']
filter_dict = dict()
for name, var in zip(filter_vars_name, filter_vars):
    filter_dict[name] = np.array(var)[NB_filt]

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

    def __init__(self, dir0, str_date, prefix='completeness_analysis.'):
        self.LOG_FILENAME = dir0 + prefix + str_date + '.log'
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
