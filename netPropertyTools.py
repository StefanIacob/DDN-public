from simulator import NetworkSimulator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import multivariate_normal
import matplotlib.patches as patches


def confidence_ellipse(cov, mean, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    cov : array-like, shape (2,2)
        covariance
    mean: : array-like, shape (2,)
        mean
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    print(ellipse)
    ax.add_patch(ellipse)


def visualize_GMM_net(pop):
    mix = pop.mix
    mu_x = pop.mu_x
    mu_y = pop.mu_y
    var_x = pop.variance_x
    var_y = pop.variance_y
    corr_xy = pop.correlation
    k = pop.k
    conn = pop.connectivity
    x_range = pop.x_range
    y_range = pop.y_range
    # mu_x_in, mu_y_in = pop.mu_in
    width_in = (x_range[1] - x_range[0])
    height_in = (y_range[1] - y_range[0])
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)
    for i in range(k):
        # plot the GMM
        N = 300
        s = 0.1
        x_extra = s * x_range[1] - x_range[0]
        y_extra = s * y_range[1] - y_range[0]

        X = np.linspace(x_range[0] - x_extra, x_range[1] + x_extra, N)
        Y = np.linspace(y_range[0] - y_extra, y_range[1] + y_extra, N)
        X, Y = np.meshgrid(X, Y)
        pos = np.dstack((X, Y))
        mx = mu_x[i]
        my = mu_y[i]
        s2x = var_x[i]
        s2y = var_y[i]
        s2 = np.array([
            [s2x, 0],
            [0, s2y]
        ])
        r = corr_xy[i]
        corr = np.array([
            [1, r],
            [r, 1]
        ])
        Sigma = np.matmul(np.matmul(s2, corr), s2)
        pi_i = mix[i]
        rv = multivariate_normal([mx, my], Sigma)
        Z = pi_i * rv.pdf(pos)
        plt.contour(X, Y, Z)
        # confidence_ellipse(Sigma, [mx, my], ax)
    X = np.linspace(x_range[0], x_range[1], N)
    Y = np.linspace(y_range[0], y_range[1], N)
    X, Y = np.meshgrid(X, Y)
    con = -0.5
    cX = np.zeros_like(X)
    cX[0, :] = con
    cX[-1, :] = con
    cX[:, 0] = con
    cX[:, -1] = con
    cY = np.zeros_like(Y)
    cY[0, :] = con
    cY[-1, :] = con
    cY[:, 0] = con
    cY[:, -1] = con
    Z = cX * cY
    ax.contour(X, Y, Z)

    # # Plot connectivity
    # mu_x_all = np.array([mu_x_in] + list(mu_x))
    # mu_y_all = np.array([mu_y_in] + list(mu_y))
    # col = cm.get_cmap('viridis')
    # plt.colorbar(cm.ScalarMappable(cmap=col))
    # style = "Simple, tail_width=1.5, head_width=5, head_length=5"
    # kw = dict(arrowstyle=style)
    # for i in range(k + 1):
    #     for j in range(k + 1):
    #         conn_frac = conn[i, j]
    #         mx_start = mu_x_all[i]
    #         my_start = mu_y_all[i]
    #         mx_end = mu_x_all[j]
    #         my_end = mu_y_all[j]
    #
    #         if i != j:
    #             a = patches.FancyArrowPatch((mx_start, my_start), (mx_end, my_end),
    #                                     connectionstyle="arc3,rad=.3", color=col(conn_frac), **kw)
    #             ax.add_patch(a)

    # ìnput
    X = np.linspace(x_range[0] - x_extra, x_range[1] + x_extra, N)
    Y = np.linspace(y_range[0] - y_extra, y_range[1] + y_extra, N)
    X, Y = np.meshgrid(X, Y)
    pos = np.dstack((X, Y))
    # mx = mu_x_in
    # my = mu_y_in
    # Sigma_in = 0.005 * (np.array(
    #     [
    #         [width_in, 0],
    #         [0, height_in]
    #     ])) ** 2
    # rv = multivariate_normal([mx, my], Sigma_in)
    # Z = rv.pdf(pos)
    # ax.contour(X, Y, Z)


def impulse_response(network, mag=10, rec_time=200, times=3, pre_pulse=10, visual=False):
    data_in = np.zeros((times * (pre_pulse + rec_time), ))
    for i in range(times):
        data_in[pre_pulse + i * rec_time] = mag
    sim = NetworkSimulator(network, warmup=0)
    if visual:
        sim.visualize(data_in)

    net_act = sim.get_network_data(data_in)
    return net_act

