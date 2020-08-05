from scipy.optimize import curve_fit


def linear(x, m, b):
    return m * x + b


def fit_sequence(x, y, index):

    params, pcov = curve_fit(linear, x[index], y[index])

    return params


def draw_linear_fit(ax, x, param, color='k'):

    y_mod = linear(x, *param)
    ax.plot(x, y_mod, color=color, linestyle='dashed')
