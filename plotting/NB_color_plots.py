from os.path import join
from os.path import dirname
from glob import glob
import ast
from collections import OrderedDict

import pandas as pd

from astropy.io import ascii as asc
from astropy.io import fits

import numpy as np
import matplotlib.pyplot as plt

from chun_codes import match_nosort

# Map number to alphabet for panel labelling
import string
d = dict(enumerate(string.ascii_lowercase, 0))

path0 = '/Users/cly/GoogleDrive/Research/NASA_Summer2015/Plots/color_plots'
NB_cat_path = '/Users/cly/data/SDF/NBcat/'

co_dir = dirname(__file__)

# Read in NBIA full catalog
NBIA_cat_file = join(NB_cat_path, 'NB_IA_emitters.allcols.fits')
NBIA_tab = fits.getdata(NBIA_cat_file)


def NB_spec_redshift(filt):
    """
    Purpose:
      Specify redshift limits for each spec-z sample

    :param filt:
    :return:

    z1, z2 : H-alpha
    z3, z4 : OIII
    z5, z6 : OII
    z7, z8 : Ly-alpha
    z9, z10 : NeIII
    """

    z_keys = ['z'+str(1+ii) for ii in range(10)]
    z_arr = []

    if filt == 'NB704':
        z_arr = [0.050, 0.100, 0.370, 0.475, 0.870, 0.910, 4.600, 4.900,
                 0.800, 0.850]
    if filt == 'NB711':
        z_arr = [0.050, 0.100, 0.375, 0.475, 0.875, 0.940, 4.650, 4.900,
                 0.800, 0.870]
    if filt == 'NB816':
        z_arr = [0.210, 0.260, 0.600, 0.700, 1.150, 1.225, 5.600, 5.800,
                 1.075, 1.150]
    if filt == 'NB921':
        z_arr = [0.385, 0.420, 0.810, 0.910, 1.460, 1.480, 6.520, 6.630,
                 0.000, 0.000]
    if filt == 'NB973':
        z_arr = [0.450, 0.520, 0.940, 0.975, 1.585, 1.620, 6.950, 7.100,
                 0.000, 0.000]

    return dict(zip(z_keys, z_arr))


def draw_color_selection_lines(filt, ax, xra, yra, old_selection=False,
                               paper=False):
    """
    Purpose:
      Draw color selection plots for each filter

    :param filt: filter name. NB704, NB711, NB816, NB921, NB973
    :param ax: matplotlib axes
    :param xra: range in x-axis
    :param yra: range in y-axis
    :param old_selection: bool to indicate to use old selection. Default: False
    :param paper: bool to indicate if plot is color. Default: False
    """

    linewidth = 1.0 if paper else 1.5

    # NB704 and NB711 emitters
    if 'NB7' in filt:
        # Excluding this diagonal line
        # x1 = np.arange(0.30, 1.20, 0.01)
        # ax.plot(x1, 1.70 * x1 + 0.0)

        # These are the color selection for H-alpha
        if old_selection:
            x2 = np.arange(-0.45, 0.3, 0.01)
            ax.plot(x2, 0.82 * x2 + 0.264, 'k--', linewidth=linewidth)
        else:
            x2 = np.arange(-0.45, 0.22, 0.01)
            ax.plot(x2, 0.84 * x2 + 0.125, 'k--', linewidth=linewidth)

        ax.plot(x2, 2.5 * x2 - 0.24, 'k--', linewidth=linewidth)

        # Color selection for other lines
        # Exclude for purpose of color selection in Shin+2020 paper
        # if filt == 'NB704':
        #    ax.plot(x1, 0.8955*x1 + 0.02533, 'b--')
        #    ax.plot([0.2136, 0.3], [0.294]*2, 'b--')
        # if filt == 'NB711':
        #    x3 = np.array([0.35, 1.2])
        #    ax.plot(x3, 0.8955*x3 - 0.0588, 'b--')
        #    ax.plot([0.1960, 0.35], [0.25]*2, 'b--')

    # NB816 emitters
    if filt == 'NB816':
        # Color selection for H-alpha
        x1 = np.arange(-0.60, 0.45, 0.01)

        if old_selection:
            ax.plot([0.45, 0.45], [0.8, 2.0], 'k--', linewidth=linewidth)
            ax.plot(x1, 2*x1 - 0.1, 'k--', linewidth=linewidth)
        else:
            ax.plot([0.45, 0.45], [0.9, 2.0], 'k--', linewidth=linewidth)
            ax.plot(x1, 2*x1 - 0.0, 'k--', linewidth=linewidth)

        # Color selection for other lines
        # Exclude this selection of weird emitters
        # x0 = [1.0, 2.0, 2.0, 1.0]
        # y0 = [1.0, 1.0, 2.0, 2.0]
        # ax.plot(x0 + [1.0], y0 + [1.0])

    # NB921 emitters
    if filt == 'NB921':
        # Color selection for H-alpha emitters
        x_val = [xra[0], 0.45]
        x1 = np.arange(x_val[0], x_val[1], 0.01)
        y1 = x1 * 1.46 + 0.58
        ax.plot(x1, y1, 'k--', linewidth=1.5)

        # Vertical dashed line
        ax.plot(np.repeat(x_val[1], 2), [max(y1), yra[1]], 'k--',
                linewidth=linewidth)

    # NB973 emitters:
    if filt == 'NB973':
        # Color selection for H-alpha emitters
        x_val = [0.18, 0.55]
        x1 = np.arange(x_val[0], x_val[1], 0.01)
        y1 = x1 * 2.423 + 0.06386
        ax.plot(x1, y1, 'k--', linewidth=1.5)

        # Vertical dashed line
        ax.plot(np.repeat(x_val[1], 2), [1.4, 3.0], 'k--',
                linewidth=linewidth)

        # Horizontal dashed line
        ax.plot([xra[0], 0.18], [0.5, 0.5], 'k--',
                linewidth=linewidth)


def read_config_file():
    """
    Purpose:
      Read in configuration file for color selection

    :return config_tab: AstroPy table
    """
    config_file = join(co_dir, 'NB_color_plot.txt')

    config_tab = asc.read(config_file, format='commented_header')

    return config_tab


def read_z_cat_file():
    """
    Purpose:
      Read in spectroscopic catalog

    :return config_tab: AstroPy table
    """

    z_cat_file = join(NB_cat_path, 'spec_match/current/NB_IA_zspec.txt')

    z_cat_tab = asc.read(z_cat_file)

    return z_cat_tab


def read_SE_file(infile):
    print("Reading : " + infile)

    SE_tab = asc.read(infile)

    mag  = SE_tab['col13']  # aperture photometry (col #13)
    dmag = SE_tab['col15']  # aperture photometry error (col #15)

    return mag, dmag


def latex_label_formatting(label):
    new_label = label.replace('R', 'R_C').replace('i', 'i^{\prime}')

    return r'${}$'.format(new_label)


def identify_good_phot(filt, mag_dict):
    mag_min = 0.1

    if filt == 'NB704' or filt == 'NB711':
        good_sigma = np.where((mag_dict['V_dmag'] <= mag_min) &
                              (mag_dict['R_dmag'] <= mag_min) &
                              (mag_dict['i_dmag'] <= mag_min))[0]

    if filt == 'NB816':
        good_sigma = np.where((mag_dict['B_dmag'] <= mag_min) &
                              (mag_dict['V_dmag'] <= mag_min) &
                              (mag_dict['R_dmag'] <= mag_min) &
                              (mag_dict['i_dmag'] <= mag_min))[0]

    if filt == 'NB921' or filt == 'NB973':
        good_sigma = np.where((mag_dict['B_dmag'] <= mag_min) &
                              (mag_dict['R_dmag'] <= mag_min) &
                              (mag_dict['i_dmag'] <= mag_min))[0]

    return good_sigma


def color_plot_generator(NB_cat_path, filt, config_tab=None,
                         z_cat_tab=None, ax=None):
    """
    Purpose:
      Generate two-color plots for include in paper (pre referee request)
      These plots illustrate the identification of NB excess emitters
      based on their primary emission line (H-alpha, [OIII], or [OII])
      in the NB filter

    :param NB_cat_path: str containing the path to NB-based SExtractor catalog
    :param filt: str containing the filter name
    :param config_tab: table from read_config_file()
    :param z_cat_tab: table from read_z_cat_file()
    :param ax: matplotlib.Axes
    """

    make_single_plot = 0
    paper = True
    if not ax:
        make_single_plot = 1
        paper = False

    # Read in NB excess emitter catalog
    search0 = join(NB_cat_path, filt, '{}emitters.fits'.format(filt))
    NB_emitter_file = glob(search0)[0]
    print("NB emitter catalog : " + NB_emitter_file)

    NB_tab = fits.getdata(NB_emitter_file)

    # Cross-match with NBIA full catalog to get spec-z
    idx1, NBIA_idx = match_nosort(NB_tab.ID, NBIA_tab[filt+'_ID'])

    # Define SExtractor photometric catalog filenames
    search0 = join(NB_cat_path, filt, 'sdf_pub2_*_{}.cat.mask'.format(filt))
    SE_files = glob(search0)

    # Remove z- or zr-band files
    SE_files = [file0 for file0 in SE_files if '_z' not in file0]

    # Read in ID's from SExtractor catalog
    SEx_ID = np.loadtxt(SE_files[0], usecols=0)

    NB_idx, SEx_idx = match_nosort(NB_tab.ID, SEx_ID)
    if NB_idx.size != len(NB_tab):
        print("Issue with table!")
        print("Exiting!")
        return

    if not config_tab:
        config_tab = read_config_file()

    if not z_cat_tab:
        z_cat_tab = read_z_cat_file()

    zspec0 = z_cat_tab['zspec0'][NBIA_idx]

    z_dict = NB_spec_redshift(filt)

    f_idx = np.where(config_tab['filter'] == filt)[0][0]  # config index

    # Read in SExtractor photometric catalogs
    mag_dict = OrderedDict()
    mag_dict['ID'] = NB_tab.ID
    mag_dict['zspec'] = zspec0

    for file0 in SE_files:
        mag, dmag = read_SE_file(file0)
        temp = file0.replace(join(NB_cat_path, filt, 'sdf_pub2_'), '')
        broad_filt = temp.replace('_{}.cat.mask'.format(filt), '')
        mag_dict[broad_filt+'_mag'] = mag[SEx_idx]
        mag_dict[broad_filt+'_dmag'] = dmag[SEx_idx]

    mag_dict['good_phot'] = np.zeros(len(NB_tab), dtype=np.int)

    dict_keys = mag_dict.keys()

    # Define broad-band colors
    if 'V_mag' in dict_keys and 'R_mag' in dict_keys:
        VR = mag_dict['V_mag'] - mag_dict['R_mag']
        mag_dict['VR'] = VR
    if 'R_mag' in dict_keys and 'i_mag' in dict_keys:
        Ri = mag_dict['R_mag'] - mag_dict['i_mag']
        mag_dict['Ri'] = Ri
    if 'B_mag' in dict_keys and 'V_mag' in dict_keys:
        BV = mag_dict['B_mag'] - mag_dict['V_mag']
        mag_dict['BV'] = BV
    if 'B_mag' in dict_keys and 'R_mag' in dict_keys:
        BR = mag_dict['B_mag'] - mag_dict['R_mag']
        mag_dict['BR'] = BR

    good_sigma = identify_good_phot(filt, mag_dict)
    mag_dict['good_phot'][good_sigma] = 1

    # Write CSV file
    df = pd.DataFrame(mag_dict)
    df_outfile = join(path0, filt+'_phot.csv')
    print("Writing : "+df_outfile)
    df.to_csv(df_outfile, index=False)

    x_title = config_tab['xtitle'][f_idx].replace('-', ' - ')
    y_title = config_tab['ytitle'][f_idx].replace('-', ' - ')
    x_title = latex_label_formatting(x_title)
    y_title = latex_label_formatting(y_title)

    xra = ast.literal_eval(config_tab['xra'][f_idx])
    yra = ast.literal_eval(config_tab['yra'][f_idx])

    if make_single_plot:
        out_pdf = join(path0, filt + '.pdf')

        fig, ax = plt.subplots()

    # Define axis to plot
    x_arr = []
    y_arr = []
    exec("x_arr = {}".format(config_tab['x_color'][f_idx]))
    exec("y_arr = {}".format(config_tab['y_color'][f_idx]))
    ax.scatter(x_arr[good_sigma], y_arr[good_sigma], marker='o', alpha=0.25,
               facecolor='black', edgecolor='none', linewidth=0.5, s=2,
               zorder=2)

    # Plot spectroscopic sample
    Ha_idx   = np.where((zspec0 >= z_dict['z1']) & (zspec0 <= z_dict['z2']))[0]
    OIII_idx = np.where((zspec0 >= z_dict['z3']) & (zspec0 <= z_dict['z4']))[0]
    OII_idx  = np.where((zspec0 >= z_dict['z5']) & (zspec0 <= z_dict['z6']))[0]

    s1 = ax.scatter(x_arr[Ha_idx], y_arr[Ha_idx], marker='o', s=5, alpha=0.5,
                    edgecolor='red', facecolor='none', linewidth=0.5,
                    zorder=3)
    s2 = ax.scatter(x_arr[OIII_idx], y_arr[OIII_idx], marker='o', s=5,
                    alpha=0.5, edgecolor='green', facecolor='none',
                    linewidth=0.5, zorder=3)
    s3 = ax.scatter(x_arr[OII_idx], y_arr[OII_idx], marker='o', s=5,
                    alpha=0.5, edgecolor='blue', facecolor='none',
                    linewidth=0.5, zorder=3)

    # Indicate dual NB704+NB921 emitters
    if filt == 'NB704' or filt == 'NB921':
        spec_name = z_cat_tab['cat_Name'][NBIA_idx]
        size = 5 if paper else 10
        dual_idx = [xx for xx in range(len(spec_name)) if
                    ('NB921' in spec_name[xx]) and ('NB704' in spec_name[xx])]
        if filt == 'NB704':
            ax.scatter(x_arr[dual_idx], y_arr[dual_idx], marker='x', s=size,
                       linewidth=0.25, alpha=0.75, color='green', zorder=1)
        if filt == 'NB921':
            ax.scatter(x_arr[dual_idx], y_arr[dual_idx], marker='x', s=size,
                       linewidth=0.25, alpha=0.75, color='red', zorder=1)

    ax.set_xlim(xra)
    ax.set_ylim(yra)

    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)

    ax.minorticks_on()  # Add minor tick marks
    ax.tick_params(which='both', direction='in')  # ticks on the inside

    draw_color_selection_lines(filt, ax, xra, yra, paper=paper)

    if paper and filt == 'NB973':
        ax.legend((s1, s2, s3), (r'H$\alpha$', '[O III]', '[O II]'),
                  loc='upper left', bbox_to_anchor=[1.5, 0.5],
                  frameon=False, handletextpad=0.1)

    if make_single_plot:
        fig.set_size_inches(8, 8)
        fig.savefig(out_pdf, bbox_inches='tight')


def generate_paper_plot():
    """
    Purpose:
      Generates 3x2 panel figure showing color selection for H-alpha emitters

    :return: PDF file generated
    """

    config_tab = read_config_file()

    z_cat_tab = read_z_cat_file()

    n_cols = 3
    n_rows = 2
    fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows)

    filters = ['NB704', 'NB711', 'NB816', 'NB921', 'NB973']

    for ii, filt in zip(range(len(filters)), filters):
        row = ii // n_cols
        col = ii % n_cols
        t_ax = ax[row][col]
        color_plot_generator(NB_cat_path, filt, config_tab=config_tab,
                             z_cat_tab=z_cat_tab, ax=t_ax)

        if 'NB7' in filt:
            t_ax.set_xlabel('')
            t_ax.set_xticklabels([])  # Remove x-tick labels

        if col == 1:
            t_ax.set_ylabel('')       # Remove label
            t_ax.set_yticklabels([])  # Remove y-tick labels
        # Change x-limit for R-i colors
        if col <= 1:
            t_ax.set_xlim(-0.45, 1.25)

        if col == 2:
            t_ax.yaxis.tick_right()
            t_ax.yaxis.set_label_position("right")

        # Top subplot panel label
        t_ax.annotate("(%s)" % d[ii], [0.025, 0.975], xycoords='axes fraction',
                      ha='left', va='top')

        # Lower right label for filter name
        ax[1][2].annotate("(%s) %s" % (d[ii], filt), [0.05, 0.70-0.1*ii],
                          xycoords='axes fraction', ha='left', va='top')

    ax[1][2].axis('off')  # Exclude bottom right panel

    plt.subplots_adjust(hspace=0.025, wspace=0.03)
    fig.set_size_inches(6.5, 4)

    out_pdf = join(path0, 'color_selection.pdf')
    fig.savefig(out_pdf, bbox_inches='tight')
