#!/usr/bin/env python3

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Plot the calibration of ImageNet processed under a pretrained ResNet-18.

This script creates a directory, "unweighted," in the working directory if the
directory does not already exist, and saves files there for the full data set:
1. full.txt -- metrics written by the function, "write_metrics," defined below
2. full.pdf -- plot of cumulative differences for the full ImageNet-1000
3. full_equiprob[nbins].txt -- metrics for nbins bins (equispaced in probs.)
   written by the function, "write_metrics2," defined below
4. full_equiprob[nbins].pdf -- reliability diagram for ImageNet-1000 with nbins
   bins (equispaced in probabilities)
5. full_equisamp[nbins].txt -- metrics for nbins bins (each containing the same
   number of observations) written by the function, "write_metrics2," defined
   below
6. full_equisamp[nbins].pdf -- reliability diagram for ImageNet-1000 with nbins
   bins (each containing the same number of observations)
7. full_ece1p.pdf -- plot of the l^1 empirical calibration errors as a function
   of nbins (the number of bins) with bins equispaced in probabilities
8. full_ece1s.pdf -- plot of the l^1 empirical calibration errors as a function
   of nbins (the number of bins) with an equal number of observations per bin
9. full_ece2p.pdf -- plot of the mean-square empirical calibration errors as a
   function of nbins (the number of bins) with bins equispaced in probabilities
10. full_ece2s.pdf -- plot of the mean-square empirical calibration errors as a
    function of nbins (the number of bins) with an equal number of observations
    per bin
Here, nbins (the number of bins) varies through the values 4, 8, 16, ..., 4096.

The script also saves similar files in the directory, "unweighted," for classes
given by the list "indices" defined below, where the numbers in list indices
refer to those within the 1000 classes of ImageNet. The files provide the same
results as for the full ImageNet-1000, but restricted to the class specified by
the filename, which starts [class number]-[class name] (the name of the class
refers to an English-language textual description). The associated filenames
follow the same convention outlined above for the full ImageNet-1000 data set:
suffixes are ".txt", ".pdf", "_equiprob[nbins].txt", "_equiprob[nbins].pdf",
"_equisamp[nbins].txt", "_equisamp[nbins].pdf", "_ece1p.pdf", "_ece1s.pdf",
"_ece2p.pdf", and "_ece2s.pdf" (with [class number]-[class name] prepended
before the suffix in order to form the complete file name). Here, nbins
(which is the number of bins) varies through the values 4, 8, 16, ..., 1024.

This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
"""

import math
import numpy as np
import os
import string
import torch
import torchvision.models as models
from torchvision import datasets, transforms
from PIL import Image
from multiprocessing import Process, Queue
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import calibration


# Set the path to the directory containing the data set for running inference.
infdir = '/datasets01/imagenet_full_size/061417/train'


def scores_and_results(output, target, probs):
    """
    Computes the scores and corresponding correctness

    Given output from a classifier, computes the score from the cross-entropy
    and checks whether the most likely class matches target.

    Parameters
    ----------
    output : array_like
        confidences (often in the form of scores prior to a softmax that would
        output a probability distribution over the classes) in classification
        of each example into every class
    target : array_like
        index of the correct class for each example
    probs : bool
        set to True if the scores being returned should be probabilities;
        set to False if the scores should be negative log-likelihoods

    Returns
    -------
    array_like
        scores (probabilities if probs is True, negative log-likelihoods
        if probs is False)
    array_like
        Boolean indicators of correctness of the classifications
    """
    argmaxs = torch.argmax(output, 1)

    results = argmaxs.eq(target)
    results = results.cpu().detach().numpy()

    scores = torch.nn.CrossEntropyLoss(reduction='none')
    scores = scores(output, argmaxs)
    if probs:
        scores = torch.exp(-scores)
    scores = scores.cpu().detach().numpy()
    # Randomly perturb the scores to ensure their uniqueness.
    perturb = np.ones(scores.size) - np.random.rand(scores.size) * 1e-8
    scores = scores * perturb
    scores = scores + np.random.rand(scores.size) * 1e-12

    return scores, results


def infer(inf_loader, model, num_batches, probs=False):
    """
    Conducts inference given a model and data loader

    Runs model on data loaded from inf_loader.

    Parameters
    ----------
    inf_loader : class
        instance of torch.utils.data.DataLoader
    model : class
        torch model
    num_batches : int
        expected number of batches to process (used only for gauging progress
        by printing this number)
    probs : bool, optional
        set to True if the scores being returned should be probabilities;
        set to False if the scores should be negative log-likelihoods

    Returns
    -------
    array_like
        scores (probabilities if probs is True, negative log-likelihoods
        if probs is False)
    array_like
        Boolean indicators of correctness of the classifications
    list
        ndarrays of indices of the examples classified into each class
        (the i'th entry of the list is an array of the indices of the examples
        from the data set that got classified into the i'th class)
    """
    model.eval()
    # Track the offset for appending indices to indicators (by default,
    # each minibatch gets indexed starting from 0, rather than offset).
    offset = 0
    indicators = [None] * 1000
    for k, (input, target) in enumerate(inf_loader):
        print(f'{k} of {num_batches} batches processed.')
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
        # Run inference.
        output = model(input_var)
        # Store the scores and results from the current minibatch,
        # and record which entries have the desired target indices.
        s, r = scores_and_results(output, target, probs)
        # Record the scores and results.
        if k == 0:
            scores = s.copy()
            results = r.copy()
        else:
            scores = np.concatenate((scores, s))
            results = np.concatenate((results, r))
        # Partition the results into the 1000 classes.
        for i in range(1000):
            inds = torch.nonzero(
                target == i, as_tuple=False).cpu().detach().numpy()
            if k == 0:
                indicators[i] = inds
            else:
                indicators[i] = np.concatenate((indicators[i], inds + offset))
        # Increment offset.
        offset += target.numel()
    print(f'{k + 1} of {num_batches} batches processed.')
    for i in range(1000):
        indicators[i] = np.squeeze(indicators[i])
    print('m = *scores.shape = {}'.format(*scores.shape))
    return scores, results, indicators


def write_metrics(filename, n, fraction, lenscale, kuiper, kolmogorov_smirnov):
    """
    Saves the provided metrics to a text file

    Writes to the text file named filename the parameters n, fraction,
    lenscale, kuiper, kolmogorov_smirnov, kuiper/lenscale, and
    kolmogorov_smirnov/lenscale.

    Parameters
    ----------
    filename : string
        name of the file in which to save the metrics
    n : int
        integer (for example, the sample size)
    fraction : float
        real number (for example, the fraction of the observations considered)
    lenscale : float
        standard deviation for normalizing kuiper and kolmogorov_smirnov
        in order to gauge statistical significance
    kuiper : float
        value of the Kuiper statistic
    kolmogorov_smirnov : float
        value of the Kolmogorov-Smirnov statistic

    Returns
    -------
    None
    """
    with open(filename, 'w') as f:
        f.write('n:\n')
        f.write(f'{n}\n')
        f.write('fraction:\n')
        f.write(f'{fraction}\n')
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


def write_metrics2(filename, n, ece1, ece2):
    """
    Saves the provided metrics to a text file

    Writes to the text file named filename the parameters n, ece1, and ece2.

    Parameters
    ----------
    filename : string
        name of the file in which to save the metrics
    n : int
        integer (for example, the sample size)
    ece1 : float
        l^1 empirical calibration error
    ece2 : float
        mean-square empirical calibration error

    Returns
    -------
    None
    """
    with open(filename, 'w') as f:
        f.write('n:\n')
        f.write(f'{n}\n')
        f.write('ece1:\n')
        f.write(f'{ece1}\n')
        f.write('ece2:\n')
        f.write(f'{ece2}\n')


def calibration_cumulative(
        r, s, majorticks, minorticks, fraction, filename='cumulative',
        title='miscalibration is the slope as a function of $k/n$'):
    """Thin wrapper around calibration.cumulative for multiprocessing"""
    kuiper, kolmogorov_smirnov, lenscale = calibration.cumulative(
        r, s, majorticks, minorticks, filename + '.pdf', title, fraction)
    write_metrics(filename + '.txt', len(s), fraction, lenscale, kuiper,
                  kolmogorov_smirnov)


def calibration_equiprob(q, r, s, nbins, filename='equiprob', n_resamp=0):
    """Thin wrapper around calibration.equiprob for multiprocessing"""
    ece1, ece2 = calibration.equiprob(r, s, nbins, filename + '.pdf', n_resamp)
    q.put({'routine': 'equiprob', 'nbins': nbins, 'ece1': ece1, 'ece2': ece2})
    write_metrics2(filename + '.txt', len(s), ece1, ece2)


def calibration_equisamp(
        q, r, s, nbins, filename='equisamp',
        title='reliability diagram (equal number of observations per bin)',
        n_resamp=0,
        xlabel='average of $\\;S_j^k\\;$ over $k = 1$, $2$, $\\dots$, $n_j$',
        ylabel='average of $\\;R_j^k\\;$ over $k = 1$, $2$, $\\dots$, $n_j$'):
    """Thin wrapper around calibration.equisamp for multiprocessing"""
    ece1, ece2 = calibration.equisamp(
        r, s, nbins, filename + '.pdf', title, n_resamp, xlabel, ylabel)
    q.put({'routine': 'equisamp', 'nbins': nbins, 'ece1': ece1, 'ece2': ece2})
    write_metrics2(filename + '.txt', len(s), ece1, ece2)


# Read the textual descriptions of the classes.
with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

# Conduct inference.
batch_size = 512
# Set the seeds for the random number generators.
torch.manual_seed(89259348)
np.random.seed(seed=3820497)
# Load the pretrained model.
resnet18 = models.resnet18(pretrained=True)
resnet18 = torch.nn.DataParallel(resnet18).cuda()
# Construct the data loader.
normalize = transforms.Normalize(
    mean=[.485, .456, .406], std=[.229, .224, .225])
inf_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(infdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
num_batches = math.ceil(1281167 / batch_size)
# Generate the scores, results, and subset indicators for ImageNet.
print('generating scores and results...')
s, r, inds = infer(inf_loader, resnet18, num_batches, probs=True)

# Create the directory "unweighted" for output, if necessary.
dir = 'unweighted'
try:
    os.mkdir(dir)
except FileExistsError:
    pass
dir = dir + '/'

# Set up for calibration.
fraction = 1
print()
print(f'fraction = {fraction}')

# Sort the scores and rearrange everything accordingly.
perm = np.argsort(s)
s = s[perm]
r = r[perm]
# Construct the inverse of the permutation perm.
iperm = np.zeros((len(perm)), dtype=np.int32)
for k in range(len(perm)):
    iperm[perm[k]] = k
for i in range(1000):
    inds[i] = np.sort(
        np.array([iperm[inds[i][k]] for k in range(len(inds[i]))]))

# Generate the calibration plots for the full ImageNet-1000 data set.
majorticks = 10
minorticks = 200
procs = []
q = Queue()
filename = dir + 'full'
print(f'creating files whose names start with {filename}...')
procs.append(Process(
    target=calibration_cumulative,
    args=(r, s, majorticks, minorticks, fraction),
    kwargs={'filename': filename}))
nbinss = [2**p for p in range(2, 13)]
for nbins in nbinss:
    procs.append(Process(
        target=calibration_equiprob, args=(q, r, s, nbins),
        kwargs={'filename': filename + '_equiprob' + str(nbins),
                'n_resamp': 20}))
    procs.append(Process(
        target=calibration_equisamp, args=(q, r, s, nbins),
        kwargs={'filename': filename + '_equisamp' + str(nbins),
                'n_resamp': 20}))
for iproc, proc in enumerate(procs):
    print(f'{iproc + 1} of {len(procs)} plots have started....')
    proc.start()
for iproc, proc in enumerate(procs):
    print(f'{iproc + 1} of {len(procs)} plots have finished....')
    proc.join()
# Read the results of the (multi)processing, remembering that the first process
# was for the routine "cumulative" (which doesn't put any returns in queue q).
d1p = {}
d2p = {}
d1s = {}
d2s = {}
for _ in range(len(procs) - 1):
    item = q.get()
    if item['routine'] == 'equiprob':
        d1p[item['nbins']] = item['ece1']
        d2p[item['nbins']] = item['ece2']
    else:
        d1s[item['nbins']] = item['ece1']
        d2s[item['nbins']] = item['ece2']
# Plot the summary statistics as a function of the number of bins.
for data in [d1p, d2p, d1s, d2s]:
    plt.figure()
    x = []
    y = []
    for key in sorted(data.keys()):
        x.append(key)
        y.append(data[key])
    plt.plot(x, y, 'k-*')
    plt.xscale('log')
    plt.xlabel('$m$ (number of bins)')
    if data is d1p or data is d1s:
        plt.ylabel('ECE$^1$')
    else:
        plt.ylabel('ECE$^2$')
    filenamed = filename + '_ece'
    if data is d1p:
        plt.title('ECE$^1$ for the standard reliability diagram')
        filenamed += '1p.pdf'
    elif data is d1s:
        plt.title('ECE$^1$ with equally many observations per bin')
        filenamed += '1s.pdf'
    elif data is d2p:
        plt.title('ECE$^2$ for the standard reliability diagram')
        filenamed += '2p.pdf'
    else:
        plt.title('ECE$^2$ with equally many observations per bin')
        filenamed += '2s.pdf'
    plt.tight_layout()
    plt.savefig(filenamed, bbox_inches='tight')
    plt.close()

# Generate the calibration plots for selected classes of the ImageNet data set,
# where indices specifies which classes, as well as for the full ImageNet-1000
# training set (corresponding to the "index" -1).
indices = [-1, 60, 68, 248, 323, 342, 837]
for i in indices:
    majorticks = 10
    minorticks = 200
    procs = []
    q = Queue()
    if i == -1:
        filename = 'full'
    else:
        filename = classes[i]
        filename = filename.translate(
            str.maketrans('', '', string.punctuation))
        filename = filename.replace(' ', '-')
    filename = dir + filename
    print(f'creating files whose names start with {filename}...')
    if i == -1:
        rs = r
        ss = s
    else:
        rs = r[inds[i]]
        ss = s[inds[i]]
    procs.append(Process(
        target=calibration_cumulative,
        args=(rs, ss, majorticks, minorticks, fraction),
        kwargs={'filename': filename}))
    if i == -1:
        nbinsmax = 13
    else:
        nbinsmax = 11
    nbinss = [2**p for p in range(2, nbinsmax)]
    for nbins in nbinss:
        procs.append(Process(
            target=calibration_equiprob, args=(q, rs, ss, nbins),
            kwargs={'filename': filename + '_equiprob' + str(nbins),
                    'n_resamp': 20}))
        procs.append(Process(
            target=calibration_equisamp, args=(q, rs, ss, nbins),
            kwargs={'filename': filename + '_equisamp' + str(nbins),
                    'n_resamp': 20}))
    for iproc, proc in enumerate(procs):
        print(f'{iproc + 1} of {len(procs)} plots have started....')
        proc.start()
    for iproc, proc in enumerate(procs):
        print(f'{iproc + 1} of {len(procs)} plots have finished....')
        proc.join()
    # Read the results of the (multi)processing, remembering that one process
    # was for routine "cumulative" (which doesn't put any returns in queue q).
    d1p = {}
    d2p = {}
    d1s = {}
    d2s = {}
    for _ in range(len(procs) - 1):
        item = q.get()
        if item['routine'] == 'equiprob':
            d1p[item['nbins']] = item['ece1']
            d2p[item['nbins']] = item['ece2']
        else:
            d1s[item['nbins']] = item['ece1']
            d2s[item['nbins']] = item['ece2']
    # Plot the summary statistics as a function of the number of bins.
    for data in [d1p, d2p, d1s, d2s]:
        plt.figure()
        x = []
        y = []
        for key in sorted(data.keys()):
            x.append(key)
            y.append(data[key])
        plt.plot(x, y, 'k-*')
        plt.xscale('log')
        plt.xlabel('$m$ (number of bins)')
        if data is d1p or data is d1s:
            plt.ylabel('ECE$^1$')
        else:
            plt.ylabel('ECE$^2$')
        filenamed = filename + '_ece'
        if data is d1p:
            plt.title('ECE$^1$ for the standard reliability diagram')
            filenamed += '1p.pdf'
        elif data is d1s:
            plt.title('ECE$^1$ with equally many observations per bin')
            filenamed += '1s.pdf'
        elif data is d2p:
            plt.title('ECE$^2$ for the standard reliability diagram')
            filenamed += '2p.pdf'
        else:
            plt.title('ECE$^2$ with equally many observations per bin')
            filenamed += '2s.pdf'
        plt.tight_layout()
        plt.savefig(filenamed, bbox_inches='tight')
        plt.close()
