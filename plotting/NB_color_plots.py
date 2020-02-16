from os.path import join
from os.path import dirname
from glob import glob
import ast

from astropy.io import ascii as asc
from astropy.io import fits

import numpy as np
import matplotlib.pyplot as plt

from chun_codes import match_nosort

path0 = '/Users/cly/GoogleDrive/Research/NASA_Summer2015/Plots/color_plots'

co_dir = dirname(__file__)


def read_config_file():
    config_file = join(co_dir, 'NB_color_plot.txt')

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

    # Read in NB excess emitter catalog
    search0 = join(NB_cat_path, filt, '{}emitters.fits'.format(filt))
    NB_emitter_file = glob(search0)[0]
    print("NB emitter catalog : " + NB_emitter_file)

    NB_tab = fits.getdata(NB_emitter_file)

    # Define SExtractor photometric catalog filenames
    search0 = join(NB_cat_path, filt, 'sdf_pub2_*_{}.cat.mask'.format(filt))
    SE_files = glob(search0)

    # Read in ID's from SExtractor catalog
    SEx_ID = np.loadtxt(SE_files[0], usecols=0)

    NB_idx, SEx_idx = match_nosort(NB_tab.ID, SEx_ID)
    if NB_idx.size != len(NB_tab):
        print("Issue with table!")
        print("Exiting!")
        return

    config_tab = read_config_file()

    f_idx = np.where(config_tab['filter'] == filt)[0][0]  # config index

    # Read in SExtractor photometric catalogs
    mag_arr = {}
    for file0 in SE_files:
        mag, dmag = read_SE_file(file0)
        temp = file0.replace(join(NB_cat_path, filt, 'sdf_pub2_'), '')
        broad_filt = temp.replace('_{}.cat.mask'.format(filt), '')
        mag_arr[broad_filt+'_mag'] = mag[SEx_idx]
        mag_arr[broad_filt+'_dmag'] = dmag[SEx_idx]

    dict_keys = mag_arr.keys()

    # Define broad-band colors
    if 'V_mag' in dict_keys and 'R_mag' in dict_keys:
        VR = mag_arr['V_mag'] - mag_arr['R_mag']
    if 'R_mag' in dict_keys and 'i_mag' in dict_keys:
        Ri = mag_arr['R_mag'] - mag_arr['i_mag']
    if 'B_mag' in dict_keys and 'V_mag' in dict_keys:
        BV = mag_arr['B_mag'] - mag_arr['V_mag']
    if 'B_mag' in dict_keys and 'R_mag' in dict_keys:
        BR = mag_arr['B_mag'] - mag_arr['R_mag']

    x_title = config_tab['xtitle'][f_idx]
    y_title = config_tab['ytitle'][f_idx]
    xra = ast.literal_eval(config_tab['xra'][f_idx])
    yra = ast.literal_eval(config_tab['yra'][f_idx])

    out_pdf = join(path0, pdf_prefix + '_' + filt + '.pdf')

    fig, ax = plt.subplots()

    # Define axis to plot
    exec("x_arr = {}".format(config_tab['x_color'][f_idx]))
    exec("y_arr = {}".format(config_tab['y_color'][f_idx]))
    ax.scatter(x_arr, y_arr, marker='o', s=5)

    ax.set_xlim(xra)
    ax.set_ylim(yra)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)

    fig.savefig(out_pdf, bbox_inches='tight')
