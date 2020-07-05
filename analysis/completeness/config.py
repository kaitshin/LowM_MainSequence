from . import lambdac, epsilon
import numpy as np

from ..NB_errors import mag_combine

# Filter name and corresponding broad-band for NB color excess plot
filters = ['NB704', 'NB711', 'NB816', 'NB921', 'NB973']
cont0 = [r'$R_Ci^{\prime}$', r'$R_Ci^{\prime}$', r'$i^{\prime}z^{\prime}$',
         r'$z^{\prime}$', r'$z^{\prime}$']

# Limiting magnitudes for NB and BB data
m_NB = np.array([26.7134 - 0.047, 26.0684, 26.9016 + 0.057, 26.7088 - 0.109, 25.6917 - 0.051])
m_BB1 = np.array([28.0829, 28.0829, 27.7568, 26.8250, 26.8250])
m_BB2 = np.array([27.7568, 27.7568, 26.8250, 00.0000, 00.0000])
cont_lim = mag_combine(m_BB1, m_BB2, epsilon)

# Minimum NB excess color for selection
minthres = [0.15, 0.15, 0.15, 0.2, 0.25]

# Prefix for mag-to-mass interpolation files
prefixes = ['Ha-NB7', 'Ha-NB7', 'Ha-NB816', 'Ha-NB921', 'Ha-NB973']

# NB statistical filter correction
filt_corr = [1.289439104, 1.41022358406, 1.29344789854,
             1.32817034288, 1.29673596942]

# For H-alpha luminosity calculation
z_NB = lambdac / 6562.8 - 1.0

# Grid definitions for log-normal EW distributions
logEW_mean_start = np.array([1.25, 1.25, 1.25, 1.25, 0.90])
logEW_sig_start = np.array([0.15, 0.55, 0.25, 0.35, 0.55])
n_mean = 4
n_sigma = 4

# Bin size for NB magnitude
NB_bin = 0.25
