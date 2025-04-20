The accompanying codes reproduce all figures and statistics presented in
the research paper, "Metrics of calibration for probabilistic predictions,"
by Imanol Arrieta-Ibarra, Paman Gujral, Jonathan Tannen, Mark Tygert, and
Cherie Xu. This repository also provides the LaTeX and BibTeX sources required
for replicating the paper.

_N.B._: The latest version of the codes calculates a version of ECEs that is
different from that in the original paper. The different version is probably
the most popular among those of the machine-learning community. To reproduce
the paper as published, please use the fourth commit, which is available on
[GitHub](https://github.com/facebookresearch/ecevecce/commit/2d413c7) ... or
for results computed using the latest version of the codes, see the talk on
[Zenodo](https://zenodo.org/records/15238204)

The main files in the repository are the following:

``tex/ecevecce.pdf``
PDF version of the paper

``tex/ecevecce.tex``
LaTeX source for the paper

``tex/ecevecce.bib``
BibTeX source for the paper

``codes/calibration.py``
Functions for plotting calibration, both cumulative and reliability diagrams

``codes/imagenetcal.py``
Python script for processing ImageNet using a pre-trained ResNet-18

``codes/imagenet_classes.txt``
Text file containing a dictionary of the names of the classes in ImageNet

``codes/pvals.py``
Python script which prints P-values for some given values of the ECCEs

``codes/dists.py``
Functions for calculating cumulative distribution functions for Brownian motion
(redistributed from [Mark Tygert's website](http://tygert.com/dists.py))

Regenerating all the figures requires running in the directory ``codes`` both
``calibration.py`` and ``imagenetcal.py``. Only ``codes/imagenetcal.py`` needs
a GPU (running on a GPU simultaneously with CPU cores); the other codes need
only CPUs.

The command-line tool ``convert`` uses [ImageMagick](https://imagemagick.org)
for conversion of PDF files to JPEG files; install ImageMagick to enable such
conversion.

********************************************************************************

Copyright license

This ecevecce software is licensed under the (MIT-type) copyright LICENSE file
in the root directory of this source tree.
