from os.path import join
from os.path import dirname
from glob import glob
from astropy.io import ascii as asc

import numpy as np
import matplotlib.pyplot as plt


path0 = '/Users/cly/GoogleDrive/Research/NASA_Summer2015/Plots/color_plots'

co_dir = dirname(__file__)


def read_config_file():
    config_file = join(co_dir, 'NBIA_color_plot.txt')

    config_tab = asc.read(config_file, format='commented_header')

    return config_tab


def read_SE_file(infile):
    print("Reading : " + infile)

    SE_tab = asc.read(infile)

    mag  = SE_tab['col13']  # aperture photometry (col #13)
    dmag = SE_tab['col15']  # aperture photometry error (col #15)

    return mag, dmag


def color_plot_generator(NB_cat_path, filt, pdf_prefix):
    """
    Purpose:
      Generate two-color plots for include in paper (pre referee request)
      These plots illustrate the identification of NB excess emitters
      based on their primary emission line (H-alpha, [OIII], or [OII])
      in the NB filter

    :param NB_cat_path: str containing the path to NB-based SExtractor catalog
    :param filt: str containing the filter name
    :param pdf_prefix: str containing the output file name
    """

    search0 = join(NB_cat_path, filt, 'sdf_pub2_*_{}.cat.mask'.format(filt))
    SE_files = glob(search0)
    print(SE_files)

    config_tab = read_config_file()

    out_pdf = join(path0, pdf_prefix + '_' + filt + '.pdf')
