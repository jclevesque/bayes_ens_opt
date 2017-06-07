# MIT License

# Copyright (c) 2017 Julien-Charles Lévesque

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

""" Wrappers for ensemble utility functions in C. """

import os

import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_double

# input type for the cos_doubles function
# must be a double array, with single dimension that is contiguous
array_2d_int = npct.ndpointer(dtype=np.int, ndim=2, flags='CONTIGUOUS')
array_1d_int = npct.ndpointer(dtype=np.int, ndim=1, flags='CONTIGUOUS')
array_1d_double = npct.ndpointer(dtype=np.float, ndim=1, flags='CONTIGUOUS')

# load the library, using numpy mechanisms
# numpy 1.7.1 and below is a mess in dealing with extension names, so I added
# this hardcoded bit for now.
from distutils.sysconfig import get_config_var
suffix = get_config_var("EXT_SUFFIX") or get_config_var("SO")

libens = npct.load_library("ens_cfuncs" + suffix, os.path.dirname(__file__))

# setup the return typs and argument types
libens.majority_vote.restype = None
libens.majority_vote.argtypes = [array_2d_int, array_1d_int, array_1d_int, c_int, c_int, c_int]

libens.weighted_vote.restype = None
libens.weighted_vote.argtypes = [array_2d_int, array_1d_double, array_1d_int, array_1d_int, c_int, c_int, c_int]

libens.diversity_kw.restype = c_double
libens.diversity_kw.argtypes = [array_2d_int, array_1d_int, c_int, c_int]

libens.diversity_entropy.restype = c_double
libens.diversity_entropy.argtypes = [array_2d_int, array_1d_int, c_int, c_int]


def majority_vote(votes):
    n, m = np.shape(votes)

    votes = np.ascontiguousarray(votes, dtype=int)
    unique_votes = np.unique(votes)
    out_predictions = np.zeros(n, dtype=int)

    libens.majority_vote(votes, unique_votes,
      out_predictions, n, m, len(unique_votes))
    return out_predictions


def weighted_vote(votes, weights):
    votes = np.ascontiguousarray(votes, dtype=int)
    n, m = np.shape(votes)

    unique_votes = np.unique(votes)
    out_predictions = np.zeros(n, dtype=int)

    libens.weighted_vote(votes,
        np.ascontiguousarray(weights, dtype=float), 
        unique_votes, out_predictions,
        n, m, len(unique_votes))
    return out_predictions


def kohavi_wolpert(votes, labels):
    n, m = np.shape(votes)
    kw = libens.diversity_kw(np.ascontiguousarray(votes), labels, n, m)
    return kw


def entropy(votes, labels):
    n, m = np.shape(votes)
    H = libens.diversity_entropy(np.ascontiguousarray(votes), labels, n, m)
    return H

