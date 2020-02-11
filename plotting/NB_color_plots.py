from os.path import join
import numpy as np
import matplotlib.pyplot as plt

path0 = '/Users/cly/GoogleDrive/Research/NASA_Summer2015/Plots/color_plots'


def color_plot_generator(filt, outfile):
    """
    Purpose:
      Generate two-color plots for include in paper (pre referee request)
      These plots illustrate the identification of NB excess emitters
      based on their primary emission line (H-alpha, [OIII], or [OII])
      in the NB filter

    :param filt: str containing the filter name
    :param outfile: str containing the output file name
    """

    out_pdf = join(path0, outfile)
