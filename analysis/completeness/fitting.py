from scipy.optimize import curve_fit


def linear(x, m, b):
    return m * x + b


def fit_sequence(x, y, index):

    params, pcov = curve_fit(linear, x[index], y[index])

    return params


def draw_linear_fit(ax, x, param, y_pos, color='k'):

    y_mod = linear(x, *param)
    ax.plot(x, y_mod, color=color, linestyle='dashed')

    ann_text = r'$\alpha$=%.2f $\gamma$=%.2f' % (param[0], param[1])
    ax.annotate(ann_text, [0.05, y_pos], va='top', ha='left',
                color=color, xycoords='axes fraction')
