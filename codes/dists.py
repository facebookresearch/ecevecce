#!/usr/bin/env python3

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Calculate cumulative distribution functions for standard Brownian motions.

Running as a script tests assertions that closed-form, analytical expressions
for the means match numerical evaluations of the means for the cumulative
distribution functions, prints values of the cumulative distribution functions
at some interesting values for their arguments, saves to disk plots in pdf
of the complementary cumulative distribution functions, and saves to disk plots
in both pdf and jpg of calibration curves for synthetic data sets drawn from
perfectly calibrated distributions. The script saves plots in the current
directory, in files named "gauss.pdf", "gauss_log.pdf", "kuiper.pdf",
"kuiper_log.pdf", "kolmogorov_smirnov.pdf", and "kolmogorov_smirnov_log.pdf".
The files whose names end with "_log.pdf" use log scales for the vertical axes.
The plots for the metrics of Kuiper and of Kolmogorov and Smirnov include
vertical dotted lines at the means associated with the corresponding
distribution. The script saves twelve other plots in the current directory,
too, as detailed in the docstring for function plotnull below.

An article detailing the functions named after mathematicians and statisticians
(Kolmogorov, Smirnov, Kuiper, Gauss, and Chebyshev) is Mark Tygert's
"Calibration of P-values for calibration and for deviation of a subpopulation
from the full population."

Functions
---------
kolmogorov_smirnov
    Evaluates the cumulative distribution function for the maximum
    of the absolute value of the standard Brownian motion on [0, 1]
kuiper
    Evaluates the cumulative distribution function for the range
    (maximum minus minimum) of the standard Brownian motion on [0, 1]
gauss
    Evaluates the cumulative distribution function for the distribution N(0, 1)
    (the standard normal distribution, involving a Gaussian)
chebyshev
    Integrates the function f(x) from x=a to x=b using n Chebyshev nodes
testmeans
    Verifies that the means of the cumulative distribution functions are right
printvals
    Evaluates the cumulative distribution functions at some points of interest
    and prints them
saveplots
    Plots and saves to disk the complementary cumulative distribution functions
plotnull
    Plots the P-values for data generated from a perfectly calibrated model

This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
"""


import math
import numpy as np
from numpy.random import default_rng
import subprocess
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def kolmogorov_smirnov(x):
    """
    Evaluates the cumulative distribution function for the maximum
    of the absolute value of the standard Brownian motion on [0, 1]

    Parameters
    ----------
    x : float
        argument at which to evaluate the cumulative distribution function
        (must be positive)

    Returns
    -------
    float
        cumulative distribution function evaluated at x
    """
    assert x > 0
    # Compute the machine precision assuming binary numerical representations.
    eps = 7 / 3 - 4 / 3 - 1
    # Determine how many terms to use to attain accuracy eps.
    fact = 4 / math.pi
    kmax = math.ceil(
        1 / 2 + x * math.sqrt(2) / math.pi * math.sqrt(math.log(fact / eps)))
    # Sum the series.
    c = 0
    for k in range(kmax):
        kplus = k + 1 / 2
        c += (-1)**k / kplus * math.exp(-kplus**2 * math.pi**2 / (2 * x**2))
    c *= 2 / math.pi
    return c


def kuiper(x):
    """
    Evaluates the cumulative distribution function for the range
    (maximum minus minimum) of the standard Brownian motion on [0, 1]

    Parameters
    ----------
    x : float
        argument at which to evaluate the cumulative distribution function
        (must be positive)

    Returns
    -------
    float
        cumulative distribution function evaluated at x
    """
    assert x > 0
    # Compute the machine precision assuming binary numerical representations.
    eps = 7 / 3 - 4 / 3 - 1
    # Determine how many terms to use to attain accuracy eps.
    fact = 4 / math.sqrt(2 * math.pi) * (1 / x + x / math.pi**2)
    kmax = math.ceil(
        1 / 2 + x / math.pi / math.sqrt(2) * math.sqrt(math.log(fact / eps)))
    # Sum the series.
    c = 0
    for k in range(kmax):
        kplus = k + 1 / 2
        c += (8 / x**2 + 2 / kplus**2 / math.pi**2) * math.exp(
            -2 * kplus**2 * math.pi**2 / x**2)
    return c


def gauss(x):
    """
    Evaluates the cumulative distribution function for the distribution N(0, 1)
    (the standard normal distribution, involving a Gaussian)

    Parameters
    ----------
    x : float
        argument at which to evaluate the cumulative distribution function

    Returns
    -------
    float
        cumulative distribution function evaluated at x
    """
    return (1 + math.erf(x / math.sqrt(2))) / 2


def chebyshev(a, b, n, f):
    """
    Integrates the function f(x) from x=a to x=b using n Chebyshev nodes

    Parameters
    ----------
    a : float
        lower limit of integration
    b : float
        upper limit of integration
    n : int
        number of Chebyshev nodes in the Gauss-Chebyshev quadrature
    f : callable
        real-valued function of a real argument to be integrated

    Returns
    -------
    float
        integral from x=a to x=b of f(x) (dx)
    """
    sum = 0
    for k in range(n):
        c = math.cos((2 * k + 1) * math.pi / (2 * n))
        x = a + (b - a) * (1 + c) / 2
        sum += f(x) * math.sqrt(1 - c**2)
    sum *= (b - a) * math.pi / (2 * n)
    return sum


def testmeans():
    """
    Verifies that the means of the cumulative distribution functions are right

    Returns
    -------
    float
        mean of the Kolmogorov-Smirnov statistic under the null hypothesis
        that the subpopulation arises from the full population's distribution
        (and that the scores are dense in their domain)
    float
        mean of the Kuiper statistic under the null hypothesis
        that the subpopulation arises from the full population's distribution
        (and that the scores are dense in their domain)

    References
    ----------
    William Feller, "The asymptotic distribution of the range of sums of
        independent random variables," Ann. Math. Statist., 22 (1951): 427-432.
    Jaume Masoliver, "Extreme values and the level-crossing problem: an
        application to the Feller process," Phys. Rev. E., 89 (2014): 042106.
    """
    # Compute the means of the Kolmogorov-Smirnov and Kuiper statistics
    # using closed-form analytic expressions (see Formula 1.4 of the reference
    # to Feller given in the docstring, as well as Formula 46 of the reference
    # to Masoliver).
    ks_mean = math.sqrt(math.pi / 2)
    ku_mean = 2 * math.sqrt(2 / math.pi)
    # Compute the means from the associated cumulative distribution functions
    # evaluated numerically.
    ks_mean2 = chebyshev(1e-8, 8, 100000, lambda x: 1 - kolmogorov_smirnov(x))
    ku_mean2 = chebyshev(1e-8, 8, 100000, lambda x: 1 - kuiper(x))
    # Check that the calculated values agree with each other.
    tolerance = 1e-8
    assert (ks_mean - ks_mean2) / ks_mean < tolerance
    assert (ku_mean - ku_mean2) / ku_mean < tolerance
    return ks_mean, ku_mean


def printvals(ks_mean, ku_mean):
    """
    Evaluates the cumulative distribution functions at some points of interest
    and prints them

    Parameters
    ----------
    ks_mean : float
        mean of the Kolmogorov-Smirnov statistic under the null hypothesis
        that the subpopulation arises from the full population's distribution
        (and that the scores are dense in their domain)
    ku_mean : float
        mean of the Kuiper statistic under the null hypothesis
        that the subpopulation arises from the full population's distribution
        (and that the scores are dense in their domain)
    """
    print(f'1 - kolmogorov_smirnov(0.001) = {1 - kolmogorov_smirnov(0.001)}')
    print('1 - kolmogorov_smirnov(ks_mean) = {}'
          .format(1 - kolmogorov_smirnov(ks_mean)))
    print('1 - kolmogorov_smirnov(9.347180056695407) = {}'
          .format(1 - kolmogorov_smirnov(9.347180056695407)))
    print('1 - kolmogorov_smirnov(4.433008036126233) = {}'
          .format(1 - kolmogorov_smirnov(4.433008036126233)))
    print('1 - kolmogorov_smirnov(2.2049236860640984) = {}'
          .format(1 - kolmogorov_smirnov(2.2049236860640984)))
    print(f'1 - kolmogorov_smirnov(1000) = {1 - kolmogorov_smirnov(1000)}')

    print()

    print(f'1 - kuiper(0.001) = {1 - kuiper(0.001)}')
    print(f'1 - kuiper(ku_mean) = {1 - kuiper(ku_mean)}')
    print(f'1 - kuiper(9.604586718869454) = {1 - kuiper(9.604586718869454)}')
    print(f'1 - kuiper(4.500374236241608) = {1 - kuiper(4.500374236241608)}')
    print(f'1 - kuiper(2.2585549672545224) = {1 - kuiper(2.2585549672545224)}')
    print(f'1 - kuiper(1000) = {1 - kuiper(1000)}')

    print()

    print('switch the mean values and see that the P-values deviate '
          + 'far from 0.5:')
    print('1 - kolmogorov_smirnov(ku_mean) = {}'
          .format(1 - kolmogorov_smirnov(ku_mean)))
    print(f'1 - kuiper(ks_mean) = {1 - kuiper(ks_mean)}')


def saveplots(ks_mean, ku_mean):
    """
    Plots and saves to disk the complementary cumulative distribution functions

    The plots, saved in the current directory, are "gauss.pdf",
    "gauss_log.pdf", "kuiper.pdf", "kuiper_log.pdf", "kolmogorov_smirnov.pdf",
    and "kolmogorov_smirnov_log.pdf". The files whose names end with "_log.pdf"
    use logarithmic scales for the vertical axes. The plots for Kuiper
    and for Kolmogorov and Smirnov include vertical dotted lines at the means
    of the corresponding distribution, assuming that the input parameters
    are correct.

    Parameters
    ----------
    ks_mean : float
        mean of the Kolmogorov-Smirnov statistic under the null hypothesis
        that the subpopulation arises from the full population's distribution
        (and that the scores are dense in their domain)
    ku_mean : float
        mean of the Kuiper statistic under the null hypothesis
        that the subpopulation arises from the full population's distribution
        (and that the scores are dense in their domain)
    """
    for func in ['gauss', 'kuiper', 'kolmogorov_smirnov']:
        for logscale in [True, False]:
            # Create a plot.
            plt.figure(figsize=[4.8, 3.6])
            # Create abscissae and ordinates.
            xmax = 8
            x = np.arange(1e-3, xmax, 1e-3)
            y = 1 - np.vectorize(globals()[func])(x)
            # Plot y versus x.
            plt.plot(x, y, 'k')
            # Plot a vertical line at the mean.
            if func == 'kuiper':
                mean = ku_mean
            elif func == 'kolmogorov_smirnov':
                mean = ks_mean
            else:
                mean = 0
            if mean > 0:
                plt.vlines(mean, 1 - globals()[func](xmax), 1, 'k', 'dotted')
                plt.text(
                    mean, 1 - globals()[func](xmax), 'mean ',
                    ha='center', va='top')
            # Set the vertical axis to use a logscale if logscale is True.
            if logscale:
                plt.yscale('log')
            # Title the axes.
            plt.xlabel('$x$')
            if func == 'kuiper':
                plt.ylabel('$1 - F(x)$')
            elif func == 'kolmogorov_smirnov':
                plt.ylabel('$1 - D(x)$')
            else:
                plt.ylabel('$1 - \\Phi(x)$')
            # Clean up the whitespace in the plot.
            plt.tight_layout()
            # Save the plot.
            filename = func
            if logscale:
                filename += '_log'
            filename += '.pdf'
            plt.savefig(filename, bbox_inches='tight')
            plt.close()


def plotnull(ns, points, transform=None, suffix=''):
    """
    Plots the P-values for data generated from a perfectly calibrated model

    The plots, saved in the current directory are "kuiper_ecdf[suffix].pdf",
    "kuiper_ecdf[suffix].jpg", "kolmogorov_smirnov_ecdf[suffix].pdf", and
    "kolmogorov_smirnov_ecdf[suffix].jpg". The JPEG versions are conversions
    from the PDF at a resolution of 1200 pixels per inch. The plots display
    the empirical cumulative distribution functions of the P-values associated
    with the Kuiper and Kolmogorov-Smirnov statistics, for points data sets,
    each with the number of scores and corresponding Bernoulli responses
    given by the given entry in ns (running everything again for each entry
    in ns). The Bernoulli responses are independent, and the probability
    of success for each is equal to the corresponding score (ensuring perfect
    calibration of the underlying data distribution). The transform gets
    applied to each score, with the scores being equispaced before application
    of transform (and remaining equispaced if transform is None).

    Parameters
    ----------
    ns : list of int
        sample sizes for each generated data set
    points : int
        number of data sets to generate per calibration curve (that is,
        per empirical cumulative distribution function of P-values)
    transform : callable, optional
        numpy function to apply to the otherwise equispaced scores
        (set to None -- the default -- to use the original equispaced scores)
    suffix : string, optional
        suffix to append to the filename (defaults to the empty string)
    """
    # Store processes for converting from pdf to jpeg in procs.
    procs = []
    # Store the calibration curves for both Kolmogorov-Smirnov and Kuiper
    # statistics (these are empirical cumulative distribution functions),
    # in ksc and kuc, respectively.
    ksc = np.zeros((len(ns), points))
    kuc = np.zeros((len(ns), points))
    for j, n in enumerate(ns):
        rng = default_rng(seed=543216789)
        # Run simulations points times.
        pks = np.zeros((points))
        pku = np.zeros((points))
        for k in range(points):
            # Generate predicted probabilities (the "scores").
            s = np.arange(0, 1, 1 / n)[:n]
            if transform is not None:
                s = transform(s)
            # Generate a sample of classifications (the "responses")
            # into two classes, correct (class 1) and incorrect (class 0),
            # avoiding numpy's random number generators that are based
            # on random bits -- they yield strange results for many seeds.
            uniform = rng.uniform(size=(n))
            r = (uniform <= s).astype(float)
            # Calculate the cumulative differences.
            c = (np.cumsum(r) - np.cumsum(s)) / n
            # Calculate the estimate of sigma.
            sigma = np.sqrt(np.sum(s * (1 - s))) / n
            # Compute the normalized Kolmogorov-Smirnov and Kuiper statistics.
            ks = np.abs(c).max() / sigma
            c = np.insert(c, 0, [0])
            ku = (c.max() - c.min()) / sigma
            # Evaluate the P-values.
            pks[k] = 1 - kolmogorov_smirnov(ks)
            pku[k] = 1 - kuiper(ku)
        # Calculate the empirical cumulative distributions of the P-values.
        ksc[j, :] = np.sort(pks)
        kuc[j, :] = np.sort(pku)
    for stat in ['kolmogorov_smirnov', 'kuiper']:
        # Create a plot.
        plt.figure(figsize=[4.8, 3.6])
        # Title the axes.
        plt.xlabel('$x$')
        plt.ylabel('fraction of P-values $\\leq x$')
        # Plot the empirical cumulative distribution functions.
        frac = np.arange(1 / points, 1 + 1 / points, 1 / points)[:points]
        for j in range(len(ns)):
            if stat == 'kolmogorov_smirnov':
                plt.plot(ksc[j, :], frac, color='k')
            elif stat == 'kuiper':
                plt.plot(kuc[j, :], frac, color='k')
        # Add a diagonal line from (0, 0) to (1, 1).
        zeroone = np.asarray((0, 1))
        plt.plot(zeroone, zeroone, 'k', linestyle='dashed')
        # Save the plot.
        filepdf = stat + '_ecdf' + suffix + '.pdf'
        plt.savefig(filepdf, bbox_inches='tight')
        plt.close()
        # Convert the pdf to jpg.
        filejpg = filepdf[:-4] + '.jpg'
        args = ['convert', '-density', '1200', filepdf, filejpg]
        procs.append(subprocess.Popen(args))
    print('waiting for conversion from pdf to jpg to finish....')
    for iproc, proc in enumerate(procs):
        proc.wait()
        print(f'{iproc + 1} of {len(procs)} conversions are done....')


if __name__ == '__main__':
    # Test if the cumulative distribution functions yield the known values
    # for their means.
    print('testing means...')
    ks_mean, ku_mean = testmeans()
    # Print values of the cumulative distribution functions at some interesting
    # values for their arguments.
    print()
    print('evaluating for particular values of the arguments...')
    print()
    printvals(ks_mean, ku_mean)
    # Save plots of the complementary cumulative distribution functions.
    print()
    print('plotting the complementary cumulative distribution functions...')
    saveplots(ks_mean, ku_mean)
    # Plot the calibration curves ("calibration curves" are the empirical
    # cumulative distribution functions of P-values under the null hypothesis
    # of perfect calibration).
    ns = [100, 1000, 10000]
    points = 100000
    print('plotting calibration with equispaced scores...')
    plotnull(ns, points)
    print('plotting calibration with square-rooted scores...')
    plotnull(ns, points, np.sqrt, suffix='_sqrt')
    print('plotting calibration with squared scores...')
    plotnull(ns, points, np.square, suffix='_square')
