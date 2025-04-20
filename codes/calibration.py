#!/usr/bin/env python3

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Calibration plots, both cumulative and two kinds of reliability diagrams

Functions
---------
cumulative
    Cumulative difference between observed and expected vals of Bernoulli vars
equiprob
    Reliability diagram with roughly equispaced average probabilities over bins
equisamp
    Reliability diagram with an equal number of observations per bin

Running as a script saves a number of plots and text files of statistics in the
subdirectory "unweighted" of the working directory, creating the subdirectory
if it does not yet exist. Many of the plots and text files get saved in
subdirectories of the subdirectory "unweighted," with each directory named
"n_nbins_inds," where n is the sample size, nbins is the number of bins used
for reliability diagrams and empirical calibration errors, and inds indexes the
three different example distributions of draws considered. Within each of these
directories named "n_nbins_inds," the script saves files

1. cumulative.pdf -- plot of empirical cumulative differences
2. cumulative_exact.pdf -- plot of cumulative differences with 0 sampling noise
3. equiprob.pdf -- reliability diagram, with bins that are equispaced in terms
                   of the scores (the scores are the predicted probabilities)
4. equisamp.pdf -- reliability diagram, with an equal number of observations
                   in every bin (except possibly for the last)
5. exact.jpg -- plot of the ideal reliability diagram with 0 sampling noise
6. metrics.txt -- summary statistics

The script also saves files directly in subdirectory "unweighted" (rather than
in the subdirectories "n_nbins_ind"), plotting the empirical calibration errors
for all directories named "n_nbins_inds" while varying nbins (and including all
3 possible values for inds simultaneously). 3 text files accompany each plot,
giving the raw data used in the 3 example distributions which inds indexes.

The other files saved in the subdirectory "unweighted" are plots of the
empirical calibration errors and empirical cumulative calibration errors for
further examples. The files ending with the suffix "_True.pdf" refer to draws
from perfectly calibrated distributions, while the files ending with the suffix
"_False.pdf" refer to draws from imperfectly calibrated distributions.
Filenames beginning with "ece1" refer to the l^1 empirical calibration error;
filenames starting with "ece2" refer to the mean-square empirical calibration;
with "kolmogorov-smirnov" refer to the empirical cumulative calibration error
based on the mean absolute deviation; and with "kuiper" refer to the empirical
cumulative calibration error based on the range (the range is the difference
between the maximum and the minimum). The next letter in each filename starting
"ece1" or "ece2" is an underscore, a "p", or an "s," where "p" refers to bins
equispaced with regard to the predicted probabilities, while "s" refers to bins
that each contain the same number of observed scores. The underscore
corresponds to plots involving multiple sample sizes. For the filenames
containing the "p" or "s" after "ece1" or "ece2", the next underscore and
following number refer to the sample size, and (after that) the next underscore
and following number refer to the an index from 0 to 2 for the 3 experiments
(when the underscore and number are present). The ".pdf" files for filenames
starting "ece1p," "ece1s," "ece2p," and "ece2s" plot all 3 experiments, while
the ".txt" files report metrics for only 1 of the 3 experiments.

This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
"""

import math
import os
import random
import re
import subprocess
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def cumulative(r, s, majorticks, minorticks, filename='cumulative.pdf',
               title='miscalibration is the slope as a function of $k/n$',
               fraction=1):
    """
    Cumulative difference between observed and expected vals of Bernoulli vars

    Saves a plot of the difference between the normalized cumulative sums of r
    and the normalized cumulative sums of s, with majorticks major ticks
    and minorticks minor ticks on the lower axis, labeling the major ticks
    with the corresponding values from s.

    Parameters
    ----------
    r : array_like
        class labels (0 for incorrect and 1 for correct classification)
    s : array_like
        success probabilities (must be unique and in strictly increasing order)
    majorticks : int
        number of major ticks on the lower axis
    minorticks : int
        number of minor ticks on the lower axis
    filename : string, optional
        name of the file in which to save the plot
    title : string, optional
        title of the plot
    fraction : float, optional
        proportion of the full horizontal axis to display

    Returns
    -------
    float
        Kuiper statistic
    float
        Kolmogorov-Smirnov statistic
    float
        quarter of the full height of the isosceles triangle
        at the origin in the plot
    """

    def histcounts(nbins, x):
        # Counts the number of entries of x
        # falling into each of nbins equispaced bins.
        j = 0
        nbin = np.zeros(nbins)
        for k in range(len(x)):
            if x[k] > (j + 1) / nbins:
                j += 1
            if j == nbins:
                break
            nbin[j] += 1
        return nbin

    assert all(s[k] < s[k + 1] for k in range(len(s) - 1))
    assert len(r) == len(s)
    plt.figure()
    ax = plt.axes()
    # Accumulate and normalize r and s.
    f = np.cumsum(r) / int(len(r) * fraction)
    ft = np.cumsum(s) / int(len(s) * fraction)
    # Plot the difference.
    plt.plot(np.insert(f - ft, 0, [0])[:(int(len(f) * fraction) + 1)], 'k')
    # Make sure the plot includes the origin.
    plt.plot(0, 'k')
    # Add an indicator of the scale of 1/sqrt(n) to the vertical axis.
    ssub = s[:int(len(s) * fraction)]
    lenscale = np.sqrt(np.sum(ssub * (1 - ssub))) / len(ssub)
    plt.plot(2 * lenscale, 'k')
    plt.plot(-2 * lenscale, 'k')
    kwargs = {
        'head_length': 2 * lenscale, 'head_width': len(ssub) / 20, 'width': 0,
        'linewidth': 0, 'length_includes_head': True, 'color': 'k'}
    plt.arrow(.1e-100, -2 * lenscale, 0, 4 * lenscale, shape='left', **kwargs)
    plt.arrow(.1e-100, 2 * lenscale, 0, -4 * lenscale, shape='right', **kwargs)
    plt.margins(x=0, y=0)
    # Label the major ticks of the lower axis with the values of s.
    ss = [
        '{:.2f}'.format(x)
        for x in np.insert(ssub, 0, [0])[::(len(ssub) // majorticks)].tolist()]
    plt.xticks(
        np.arange(majorticks) * len(ssub) // majorticks, ss[:majorticks])
    if len(ssub) >= 300 and minorticks >= 50:
        # Indicate the distribution of s via unlabeled minor ticks.
        plt.minorticks_on()
        ax.tick_params(which='minor', axis='x')
        ax.tick_params(which='minor', axis='y', left=False)
        ax.set_xticks(np.cumsum(histcounts(minorticks, ssub)), minor=True)
    # Label the axes.
    plt.xlabel('$S_k$')
    plt.ylabel('$C_k$')
    plt.twiny()
    plt.xlabel('$k/n$')
    # Title the plot.
    plt.title(title)
    # Clean up the whitespace in the plot.
    plt.tight_layout()
    # Save the plot.
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    # Calculate summary statistics.
    fft = np.insert(f - ft, 0, [0])[:(int(len(f) * fraction) + 1)]
    kuiper = np.max(fft) - np.min(fft)
    kolmogorov_smirnov = np.max(np.abs(fft))
    return kuiper, kolmogorov_smirnov, lenscale


def equiprob(r, s, nbins, filename='equiprob.pdf', n_resamp=0):
    """
    Reliability diagram with roughly equispaced average probabilities over bins

    Plots a reliability diagram with roughly equispaced average probabilities
    for the bins.

    Parameters
    ----------
    r : array_like
        class labels (0 for incorrect and 1 for correct classification)
    s : array_like
        success probabilities (must be in non-decreasing order)
    nbins : int
        number of bins
    filename : string, optional
        name of the file in which to save the plot
    n_resamp : int, optional
        number of times to resample and plot an extra line for error bars

    Returns
    -------
    float
        l^1 empirical calibration error
    float
        mean-square empirical calibration error
    """

    def bintwo(nbins, a, b, q):
        # Counts the number of entries of q falling into each of nbins
        # equispaced bins and calculates the averages per bin of the arrays
        # a and b, returning np.nan as the "average" for any bin that is empty.
        j = 0
        bina = np.zeros(nbins)
        binb = np.zeros(nbins)
        nbin = np.zeros(nbins)
        for k in range(len(q)):
            if q[k] > (j + 1) / nbins:
                j += 1
            if j == nbins:
                break
            bina[j] += a[k]
            binb[j] += b[k]
            nbin[j] += 1
        # Normalize the sum for each bin to compute the arithmetic average.
        bina = np.divide(bina, nbin, where=nbin != 0)
        bina[np.where(nbin == 0)] = np.nan
        binb = np.divide(binb, nbin, where=nbin != 0)
        binb[np.where(nbin == 0)] = np.nan
        return nbin, bina, binb

    assert all(s[k] <= s[k + 1] for k in range(len(s) - 1))
    plt.figure()
    for _ in range(n_resamp):
        # Resample from s, preserving the pairing of s with r.
        sr = np.asarray(
            [[s[k], r[k]] for k in np.random.randint(0, len(s), (len(s)))])
        perm = np.argsort(sr[:, 0])
        ss = sr[perm, 0]
        rs = sr[perm, 1]
        _, binrs, binss = bintwo(nbins, rs, ss, ss)
        # Use the light gray, "gainsboro".
        plt.plot(binss, binrs, 'gainsboro')
    nbin, binr, bins = bintwo(nbins, r, s, s)
    # Use the solid black, "k".
    plt.plot(bins, binr, 'k*:')
    zeroone = np.asarray((0, 1))
    plt.plot(zeroone, zeroone, 'k')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('average of $\\;S_j^k\\;$ over $k = 1$, $2$, $\\dots$, $n_j$')
    plt.ylabel('average of $\\;R_j^k\\;$ over $k = 1$, $2$, $\\dots$, $n_j$')
    plt.title('reliability diagram')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    # Calculate summary statistics.
    babs = np.abs(binr - bins)
    babs[np.where(np.isnan(babs))] = 0
    ece1 = np.sum(nbin * babs / len(r))
    bsqr = np.square(binr - bins)
    bsqr[np.where(np.isnan(bsqr))] = 0
    ece2 = np.sum(nbin * bsqr / len(r))
    return ece1, ece2


def equisamp(
        r, s, nbins, filename='equisamp.pdf',
        title='reliability diagram (equal number of observations per bin)',
        n_resamp=0,
        xlabel='average of $\\;S_j^k\\;$ over $k = 1$, $2$, $\\dots$, $n_j$',
        ylabel='average of $\\;R_j^k\\;$ over $k = 1$, $2$, $\\dots$, $n_j$'):
    """
    Reliability diagram with an equal number of observations per bin

    Plots a reliability diagram with an equal number of observations per bin.

    Parameters
    ----------
    r : array_like
        class labels (0 for incorrect and 1 for correct classification)
    s : array_like
        success probabilities (must be in non-decreasing order)
    nbins : int
        number of bins
    filename : string, optional
        name of the file in which to save the plot
    title : string, optional
        title of the plot
    n_resamp : int, optional
        number of times to resample and plot an extra line for error bars
    xlabel : string, optional
        label for the horizontal axis
    ylabel : string, optional
        label for the vertical axis

    Returns
    -------
    float
        l^1 empirical calibration error
    float
        mean-square empirical calibration error
    """

    def hist(a, nbins):
        # Calculates the average of a in nbins bins,
        # each containing len(a) // nbins entries of a
        # (except perhaps for the last bin)
        ns = len(a) // nbins
        hists = np.sum(np.reshape(a[:nbins * ns], (nbins, ns)), axis=1) / ns
        last = a[(nbins - 1) * ns:]
        hists[-1] = np.sum(last) / len(last)
        return hists

    assert all(s[k] <= s[k + 1] for k in range(len(s) - 1))
    plt.figure()
    for _ in range(n_resamp):
        # Resample from s, preserving the pairing of s with r.
        sr = np.asarray(
            [[s[k], r[k]] for k in np.random.randint(0, len(s), (len(s)))])
        perm = np.argsort(sr[:, 0])
        ss = sr[perm, 0]
        rs = sr[perm, 1]
        binrs = hist(rs, nbins)
        binss = hist(ss, nbins)
        # Use the light gray, "gainsboro".
        plt.plot(binss, binrs, 'gainsboro')
    binr = hist(r, nbins)
    bins = hist(s, nbins)
    # Use the solid black, "k".
    plt.plot(bins, binr, 'k*:')
    zeroone = np.asarray((0, 1))
    plt.plot(zeroone, zeroone, 'k')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    # Calculate summary statistics.
    ns = len(s) // nbins
    props = np.ones((nbins)) * ns / len(s)
    props[-1] = (len(s) - ((nbins - 1) * ns)) / len(s)
    ece1 = np.sum(props * np.abs(binr - bins))
    ece2 = np.sum(props * np.square(binr - bins))
    return ece1, ece2


def devils(j, k, places=7):
    """
    Evaluates an analogue of the devil's staircase of Cantor

    Parameters
    ----------
    j : integer
        index in a strange fractal ordering of k equispaced abscissae;
        sorting all k ordinates yields the usual ordering on the real line,
        since the devil's staircase is monotone (non-decreasing)
    k : integer
        largest possible value for j
    places : integer, optional
        bits in the mantissa of the returned ordinate
        (the ordinate lies in the unit interval [0, 1])

    Returns
    -------
    float
        ordinate j (out of k), evaluated to places bits of precision
    """
    s = '{0:o}'.format(round(j * 8**places / k))
    s = re.sub('[1-7]', '1', s)
    a = int(s, 2) / 2**places / (1 - 1 / (2**places))
    return a


if __name__ == '__main__':

    # Set parameters.
    # minorticks is the number of minor ticks on the lower axis.
    minorticks = 100
    # majorticks is the number of major ticks on the lower axis.
    majorticks = 10

    # Calculate empirical calibration errors (ECEs)
    # and the cumulative statistics for some simple examples.
    #
    # Set the number of draws per bin to be the same value for each bin.
    draws_per_bin = 2**4
    # Consider both perfectly and imperfectly calibrated distributions.
    indist = [True, False]
    # Initialize storage for the ECEs.
    ece1 = {}
    ece2 = {}
    # Initialize storage for the raw cumulative statistics.
    ku = {}
    ks = {}
    # Initialize storage for the normalized cumulative statistics.
    kunorm = {}
    ksnorm = {}
    # Loop through several sample sizes.
    log2 = round(math.log(draws_per_bin) / math.log(2))
    for n in [2**k for k in range(3 * log2, 22)]:
        print(f'processing sample size n = {n}...')
        ece1[n] = {}
        ece2[n] = {}
        ku[n] = {}
        ks[n] = {}
        kunorm[n] = {}
        ksnorm[n] = {}

        # Set the number nbins of bins.
        nbins = n // draws_per_bin
        # Construct predicted success probabilities.
        sl = np.arange(0, 1, 1 / n) + 1 / (2 * n)
        # ss is a list of predicted probabilities for 3 kinds of examples.
        ss = [sl, np.square(sl), np.sqrt(sl)]
        # Loop through the different kinds of examples.
        for inds, s in enumerate(ss):
            # The success probabilities must be in non-decreasing order.
            s = np.sort(s)

            # Set a directory for saving plots
            # (creating the directory if necessary).
            dir = 'unweighted'
            try:
                os.mkdir(dir)
            except FileExistsError:
                pass
            dir = dir + '/'

            ece1[n][inds] = {}
            ece2[n][inds] = {}
            ku[n][inds] = {}
            ks[n][inds] = {}
            kunorm[n][inds] = {}
            ksnorm[n][inds] = {}
            # Consider both perfectly and imperfectly calibrated data.
            for ind in indist:
                # Construct true underlying probabilities for sampling.
                random.seed(987654321)
                if ind:
                    t = 0
                else:
                    t = [devils(j, n) for j in range(n)]
                    t.sort()
                    t = [t[j] - 1 + math.exp(-math.exp(2) * j / n)
                         for j in range(n)]
                # Average together several (the number being montecarlo)
                # independent realizations, computing statistics for each.
                montecarlo = 9
                ece1[n][inds][ind] = 0
                ece2[n][inds][ind] = 0
                ku[n][inds][ind] = 0
                ks[n][inds][ind] = 0
                kunorm[n][inds][ind] = 0
                ksnorm[n][inds][ind] = 0
                for _ in range(montecarlo):
                    # Generate a sample of classifications into two classes,
                    # correct (class 1) and incorrect (class 0),
                    # avoiding numpy's random number generators
                    # that are based on random bits --
                    # they yield strange results for many seeds.
                    uniform = np.asarray([random.random() for _ in range(n)])
                    r = (uniform <= s + t).astype(float)
                    # Calculate the statistics.
                    filename = dir + 'cumulative.pdf'
                    kuiper, kolmogorov_smirnov, lenscale = cumulative(
                        r, s, majorticks, minorticks, filename)
                    ku[n][inds][ind] += kuiper
                    ks[n][inds][ind] += kolmogorov_smirnov
                    kunorm[n][inds][ind] += kuiper / lenscale
                    ksnorm[n][inds][ind] += kolmogorov_smirnov / lenscale
                    filename = dir + 'equisamp.pdf'
                    e1, e2 = equisamp(r, s, nbins, filename)
                    ece1[n][inds][ind] += e1
                    ece2[n][inds][ind] += e2
                ece1[n][inds][ind] /= montecarlo
                ece2[n][inds][ind] /= montecarlo
                ku[n][inds][ind] /= montecarlo
                ks[n][inds][ind] /= montecarlo
                kunorm[n][inds][ind] /= montecarlo
                ksnorm[n][inds][ind] /= montecarlo
    # Remove unneeded (unwanted?) plots.
    args = ['rm', dir + 'cumulative.pdf']
    subprocess.Popen(args)
    args = ['rm', dir + 'equisamp.pdf']
    subprocess.Popen(args)
    # Save plots with the ECEs and cumulative statistics.
    for data in [ece1, ece2, ku, ks, kunorm, ksnorm]:
        for ind in indist:
            plt.figure()
            filename = 'unweighted/'
            if data is ece1:
                filename += 'ece1'
                plt.ylabel('average ECE$^1$')
            elif data is ece2:
                filename += 'ece2'
                plt.ylabel('average ECE$^2$')
            elif data is ku:
                filename += 'kuiper'
                plt.ylabel('average ECCE-R')
            elif data is ks:
                filename += 'kolmogorov-smirnov'
                plt.ylabel('average ECCE-MAD')
            elif data is kunorm:
                filename += 'kuiper_normalized'
                plt.ylabel(r'average ECCE-R / $\sigma_n$')
            else:
                filename += 'kolmogorov-smirnov_normalized'
                plt.ylabel(r'average ECCE-MAD / $\sigma_n$')
            filename += '_' + str(ind) + '.pdf'
            plt.xscale('log')
            plt.xlabel('$n$ (sample size)')
            for inds in data[min(data)].keys():
                x = data.keys()
                y = [data[k][inds][ind] for k in data.keys()]
                if inds == 0:
                    plt.plot(x, y, 'k-*', label='equispaced')
                elif inds == 1:
                    plt.plot(x, y, 'k--*', label='squared')
                else:
                    plt.plot(x, y, 'k:*', label='square rooted')
            plt.ylim(bottom=0)
            plt.legend()
            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight')
            plt.close()

    # Generate directories with plots as specified via the code below,
    # with each directory named "n_nbins_inds," where n is the sample size,
    # nbins is the number of bins for the reliability diagrams and ECEs,
    # and inds is an index -- 0, 1, or 2 -- for the 3 examples considered.
    # Also generate plots varying nbins (saved in the directory "unweighted"
    # directly, rather than a subdirectory), which include all 3 experiments
    # corresponding to the different possible values of inds, simultaneously,
    # together with a separate text file of data for each experiment.
    #
    # Set nbinss to be the numbers of bins to consider.
    nbinss = [2**p for p in range(2, 13)]
    # Store processes for converting from pdf to jpeg in procs.
    procs = []
    # Store filenames to remove.
    to_remove = []
    # Store empirical calibration errors for each pair of n and nbins.
    ece1ps = {}
    ece2ps = {}
    ece1ss = {}
    ece2ss = {}
    # n is the number of observations (that is, the sample size).
    for n in [2**13, 2**15, 2**17]:
        ece1ps[n] = {}
        ece2ps[n] = {}
        ece1ss[n] = {}
        ece2ss[n] = {}
        # nbins is the number of bins for the reliability diagrams.
        for nbins in nbinss:
            # Construct predicted success probabilities.
            sl = np.arange(0, 1, 1 / n) + 1 / (2 * n)
            # ss is a list of predicted probabilities for 3 kinds of examples.
            ss = [sl, np.square(sl), np.sqrt(sl)]
            for inds, s in enumerate(ss):
                if inds not in ece1ps[n]:
                    ece1ps[n][inds] = {}
                if inds not in ece2ps[n]:
                    ece2ps[n][inds] = {}
                if inds not in ece1ss[n]:
                    ece1ss[n][inds] = {}
                if inds not in ece2ss[n]:
                    ece2ss[n][inds] = {}

                # The success probabilities must be in non-decreasing order.
                s = np.sort(s)

                # Construct true underlying probabilities for sampling.
                random.seed(987654321)
                t = [devils(j, n) for j in range(n)]
                t.sort()
                t = [t[j] - 1 + math.exp(-math.exp(2) * j / n)
                     for j in range(n)]

                # Set a unique directory for each collection of experiments
                # (creating the directory if necessary).
                dir = 'unweighted'
                try:
                    os.mkdir(dir)
                except FileExistsError:
                    pass
                dir = 'unweighted/' + str(n) + '_' + str(nbins)
                dir = dir + '_' + str(inds)
                try:
                    os.mkdir(dir)
                except FileExistsError:
                    pass
                dir = dir + '/'

                # Generate a sample of classifications into two classes,
                # correct (class 1) and incorrect (class 0),
                # avoiding numpy's random number generators
                # that are based on random bits --
                # they yield strange results for many seeds.
                random.seed(987654321)
                uniform = np.asarray([random.random() for _ in range(n)])
                r = (uniform <= s + t).astype(float)

                print(f'./{dir} is under construction....')

                # Generate five plots and a text file reporting metrics.
                filename = dir + 'cumulative.pdf'
                kuiper, kolmogorov_smirnov, lenscale = cumulative(
                    r, s, majorticks, minorticks, filename)
                filename = dir + 'equiprob.pdf'
                ece1p, ece2p = equiprob(r, s, nbins, filename, n_resamp=20)
                ece1ps[n][inds][nbins] = ece1p
                ece2ps[n][inds][nbins] = ece2p
                filename = dir + 'equisamp.pdf'
                ece1s, ece2s = equisamp(r, s, nbins, filename, n_resamp=20)
                ece1ss[n][inds][nbins] = ece1s
                ece2ss[n][inds][nbins] = ece2s
                filename = dir + 'metrics.txt'
                with open(filename, 'w') as f:
                    f.write('n:\n')
                    f.write(f'{n}\n')
                    f.write('lenscale:\n')
                    f.write(f'{lenscale}\n')
                    f.write('Kuiper:\n')
                    f.write(f'{kuiper:.4}\n')
                    f.write('Kolmogorov-Smirnov:\n')
                    f.write(f'{kolmogorov_smirnov:.4}\n')
                    f.write('Kuiper / lenscale:\n')
                    f.write(f'{(kuiper / lenscale):.4}\n')
                    f.write('Kolmogorov-Smirnov / lenscale:\n')
                    f.write(f'{(kolmogorov_smirnov / lenscale):.4}\n')
                    f.write(
                        'ECE1 with bins equispaced on the x-axis:\n')
                    f.write(f'{ece1p:.4}\n')
                    f.write(
                        'ECE2 with bins equispaced on the x-axis:\n')
                    f.write(f'{ece2p:.4}\n')
                    f.write(
                        'ECE1 with an equal number of scores per bin:\n')
                    f.write(f'{ece1s:.4}\n')
                    f.write(
                        'ECE2 with an equal number of scores per bin:\n')
                    f.write(f'{ece2s:.4}\n')
                filename = dir + 'cumulative_exact.pdf'
                _, _, _ = cumulative(
                    s + t, s, majorticks, minorticks, filename,
                    title='exact expectations')
                filename = dir + 'exact.pdf'
                equisamp(
                    s + t, s, n, filename, title='exact expectations',
                    xlabel='score $S_j^k$',
                    ylabel='expected value of response $R_j^k$')
                args = ['convert', '-density', '1200', filename,
                        filename[:-4] + '.jpg']
                procs.append(subprocess.Popen(args))
                to_remove.append(filename)
    # Save plots and text files with the empirical calibration errors.
    for data in [ece1ps, ece2ps, ece1ss, ece2ss]:
        for n, ecen in data.items():
            plt.figure()
            for inds, ece in ecen.items():
                filename = 'unweighted/'
                if data is ece1ps:
                    filename += 'ece1p_'
                elif data is ece1ss:
                    filename += 'ece1s_'
                elif data is ece2ps:
                    filename += 'ece2p_'
                else:
                    filename += 'ece2s_'
                filename += str(n) + '_' + str(inds) + '.txt'
                with open(filename, 'w') as f:
                    for nbins, val in ece.items():
                        f.write(f'{nbins}: {val}\n')
                if inds == 0:
                    plt.plot(
                        ece.keys(), ece.values(), 'k-*', label='equispaced')
                elif inds == 1:
                    plt.plot(
                        ece.keys(), ece.values(), 'k--*', label='squared')
                else:
                    plt.plot(
                        ece.keys(), ece.values(), 'k:*', label='square rooted')
            plt.legend()
            plt.xscale('log')
            plt.xlabel('$m$ (number of bins)')
            if data is ece1ps or data is ece1ss:
                plt.ylabel('ECE$^1$')
            else:
                plt.ylabel('ECE$^2$')
            filename = 'unweighted/'
            if data is ece1ps:
                plt.title('ECE$^1$ for the standard reliability diagram')
                filename += 'ece1p_'
            elif data is ece1ss:
                plt.title('ECE$^1$ with equally many observations per bin')
                filename += 'ece1s_'
            elif data is ece2ps:
                plt.title('ECE$^2$ for the standard reliability diagram')
                filename += 'ece2p_'
            else:
                plt.title('ECE$^2$ with equally many observations per bin')
                filename += 'ece2s_'
            filename += str(n) + '.pdf'
            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
    print('waiting for conversion from pdf to jpg to finish....')
    for iproc, proc in enumerate(procs):
        proc.wait()
        print(f'{iproc + 1} of {len(procs)} conversions are done....')
    print('deleting the pdf versions of files converted to jpg....')
    for filename in to_remove:
        args = ['rm', filename]
        subprocess.Popen(args)
