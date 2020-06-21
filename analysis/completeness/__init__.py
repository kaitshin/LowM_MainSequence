import numpy as np
import matplotlib.pyplot as plt

import logging

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

formatter = logging.Formatter('%(asctime)s - %(module)12s.%(funcName)20s - %(levelname)s: %(message)s')


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
