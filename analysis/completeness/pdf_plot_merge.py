from os.path import join, exists

import numpy as np

from astropy.io import ascii as asc

# Grid definition for log-normal distribution
from .config import logEW_mean_start, logEW_sig_start, n_mean, n_sigma
from .config import filters

from PyPDF2 import PdfFileWriter, PdfFileReader


best_file_filename = 'best_fit_completeness_50.tbl'

def merge_final_plots(dir0, filt, best_tab=None):
    """
    Purpose:
      Extract best fit results for each filter

    :param dir0: relative/absolute path for parent directobry of plots
    :param best_tab: astropy table containing best-fit
    :param filt: str. e.g., 'NB704', 'NB711', 'NB816', 'NB921', 'NB973'

    """

    # Get filter index
    filt_ii = [ff for ff in range(len(filters)) if filters[ff] == filt][0]

    # Read in best-fit file if not provided
    if isinstance(best_tab, type(None)):
        best_fit_file = join(dir0, best_file_filename)
        if not exists(best_fit_file):
            raise IOError("WARNING - File not found: " + best_fit_file)

        print("Reading : " + best_fit_file)
        best_tab = asc.read(best_fit_file)

    best_log_EWmean = best_tab['log_EWmean'][filt_ii]
    best_log_EWsig  = best_tab['log_EWsig'][filt_ii]

    suffix = ['', '.comp', '.stats', '.crop']
    n_pages = [1, 2, 1, 3]  # Number of pages for each set to get best fit

    pdf_files = [join(dir0, 'ew_MC_'+filt+s+'.pdf') for s in suffix]

    logEW_mean = logEW_mean_start[filt_ii] + 0.1 * np.arange(n_mean)
    logEW_sig = logEW_sig_start[filt_ii] + 0.1 * np.arange(n_sigma)

    mean_idx = np.where(np.absolute(logEW_mean - best_log_EWmean) < 0.01)[0][0]
    sig_idx  = np.where(np.absolute(logEW_sig - best_log_EWsig) < 0.01)[0][0]

    # This is the best-fit reference index. (counts from zero)
    # It's the starting point depending on the number of pages for each fit.
    ref_index = mean_idx * n_sigma + sig_idx

    pdf_writer = PdfFileWriter()

    for pdf_file, n_page in zip(pdf_files, n_pages):
        pdf = PdfFileReader(pdf_file)

        if '.stats.pdf' not in pdf_file:
            for nn in range(n_page):
                pdf_writer.addPage(pdf.getPage(ref_index * n_page + nn))
        else:
            pdf_writer.addPage(pdf.getPage(mean_idx))

    outfile = join(dir0, filt + '_best_fit_plots.pdf')
    print("Writing : " + outfile)
    pdf_output = open(outfile, 'wb')
    pdf_writer.write(pdf_output)


def run_merge_final_plots(dir0):
    """
    Purpose:
      Call merge_final_plots() for all NB filters

    :param dir0:
    :return:  relative/absolute path for parent directory of plots
    """

    # Read in best-fit file
    best_fit_file = join(dir0, best_file_filename)
    if not exists(best_fit_file):
        raise IOError("WARNING - File not found: " + best_fit_file)

    print("Reading : " + best_fit_file)
    best_tab = asc.read(best_fit_file)

    for filt in filters:
        merge_final_plots(dir0, filt, best_tab=best_tab)

    print("Exiting run_merge_final_plots")
