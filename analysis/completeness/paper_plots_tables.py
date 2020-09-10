import numpy as np
from collections import OrderedDict
from os.path import join

from astropy.io import ascii as asc
from astropy.table import Table

from PyPDF2 import PdfFileWriter, PdfFileReader

from .config import filters, minthres, z_NB, m_NB, filter_dict, cont_lim
from .properties import compute_EW
from .config import path0
from ..NB_errors import ew_flux_dual
from .select import color_cut

delta_NB = 0.01


def make_table(mylog, date_folder='', comp_tab=None):
    """
    Purpose:
      Generate table for paper. This is specifically Table 5

    :param mylog: MLog logging class
    :param date_folder: relative sub-folder.
      Recommended format: 3-char month and two-digit date. e.g., "sep15"
    :param comp_tab: Astropy table containing best fits
    """

    MC_folder_path = join(path0, "Completeness", date_folder)

    EW_min = np.zeros(len(filters))
    Flux_limit = np.zeros(len(filters))

    NBmin = 20.0
    for ff in range(len(filters)):
        EW_min[ff] = 10**(compute_EW(minthres[ff], ff))

        NBmax = m_NB[ff] - delta_NB
        NB = np.arange(NBmin, NBmax + delta_NB, delta_NB)

        filt_dict = {'dNB': filter_dict['dNB'][ff],
                     'dBB': filter_dict['dBB'][ff],
                     'lambdac': filter_dict['lambdac'][ff]}
        y_cut = color_cut(NB, m_NB[ff], cont_lim[ff])

        EW, NB_flux = ew_flux_dual(NB, NB+y_cut, y_cut, filt_dict)
        Flux_limit[ff] = np.log10(np.nanmin(NB_flux))

    keys   = ['filters', 'minthres', 'EW_min', 'rest_EW_min', 'Flux_limit']
    values = [filters, minthres, EW_min, EW_min/(1+z_NB), Flux_limit]
    tab_format = ['%5s', '%4.2f', '%3.1f', '%3.1f', '%4.2f']

    # Add in best-fit completeness numbers
    if isinstance(comp_tab, type(None)):
        completeness_file = join(MC_folder_path, 'best_fit_completeness_50.tbl')
        mylog.info("Reading : " + completeness_file)
        comp_tab = asc.read(completeness_file)

    keys += ['log_EWmean', 'log_EWsig', 'comp_50_sSFR', 'comp_50_SFR']
    values += [comp_tab['log_EWmean'].data,
               comp_tab['log_EWsig'].data,
               comp_tab['comp_50_sSFR'].data,
               comp_tab['comp_50_SFR'].data]
    tab_format += ['%4.2f', '%4.2f', '%.2f', '%.2f']

    table_dict = OrderedDict(zip(keys, values))

    tab0 = Table(table_dict)

    out_table_file = join(MC_folder_path, 'completeness_table.tex')
    mylog.info("Writing : " + out_table_file)
    asc.write(tab0, out_table_file, format='latex', overwrite=True,
              formats=dict(zip(keys, tab_format)))


def make_plots(mylog, date_folder=''):
    """
    Purpose:
      Generate plots for paper. These are the two panels in Figure 12
      for NB921 best fit

    :param mylog: MLog logging class
    :param date_folder: relative sub-folder.
      Recommended format: 3-char month and two-digit date. e.g., "sep15"
    """

    MC_folder_path = join(path0, "Completeness", date_folder)

    pdf_writer1 = PdfFileWriter()
    pdf_writer2 = PdfFileWriter()

    pdf_file = join(MC_folder_path, 'NB921_best_fit_plots.pdf')
    mylog.info("Reading : " + pdf_file)
    pdf = PdfFileReader(pdf_file)

    # Color excess plot
    pdf_writer1.addPage(pdf.getPage(4))

    outfile1 = join(MC_folder_path, 'comp_select.pdf')
    mylog.info("Writing : " + outfile1)
    pdf_output1 = open(outfile1, 'wb')
    pdf_writer1.write(pdf_output1)
    pdf_output1.close()

    # SFR completeness plot
    pdf_writer2.addPage(pdf.getPage(6))

    outfile2 = join(MC_folder_path, 'comp_SFR.pdf')
    mylog.info("Writing : " + outfile2)
    pdf_output2 = open(outfile2, 'wb')
    pdf_writer2.write(pdf_output2)
    pdf_output2.close()
