#!/usr/bin/env python3

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Print P-values for some given values of the cumulative statistics.

This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
"""

from dists import kolmogorov_smirnov, kuiper


# Specify the values of the normalized Kolmogorov-Smirnov statistic.
ks = [5.512, 6.607, 5.446, 4.274, 10.14, 8.004, 111.7]
# Specify the values of the normalized Kuiper statistic.
ku = [10.16, 6.780, 8.267, 5.186, 10.23, 8.008, 111.7]


if __name__ == '__main__':
    for val in ks:
        print(f'1 - kolmogorov_smirnov({val}) = {1 - kolmogorov_smirnov(val)}')
    for val in ku:
        print(f'1 - kuiper({val}) = {1 - kuiper(val)}')
