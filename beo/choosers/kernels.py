##
# Copyright (C) 2012 Jasper Snoek, Hugo Larochelle and Ryan P. Adams
#
# This code is written for research and educational purposes only to
# supplement the paper entitled
# "Practical Bayesian Optimization of Machine Learning Algorithms"
# by Snoek, Larochelle and Adams
# Advances in Neural Information Processing Systems, 2012
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

SQRT_3 = np.sqrt(3.0)
SQRT_5 = np.sqrt(5.0)


def dist2(ls, x1, x2=None):
    # Assumes NxD and MxD matrices.
    # Compute the squared distance matrix, given length scales.
    if ls.shape[0] > x1[0].shape[0]:
        ls = ls[:-1]

    if x2 is None:
        # Find distance with self for x1
        # Rescale.
        xx1 = x1 / ls
        xx2 = xx1

    else:
        # Rescale.
        xx1 = x1 / ls
        xx2 = x2 / ls

    r2 = np.maximum(-(2 * np.dot(xx1, xx2.T)
                       - np.sum(xx1*xx1, axis=1)[:,np.newaxis]
                       - np.sum(xx2*xx2, axis=1)[:,np.newaxis].T), 0.0)
    return r2


def grad_dist2(ls, x1, x2=None):
    if ls.shape[0] > x1[0].shape[0]:
        ls = ls[:-1]

    if x2 is None:
        x2 = x1

    # Rescale.
    x1 = x1 / ls
    x2 = x2 / ls

    N = x1.shape[0]
    M = x2.shape[0]
    D = x1.shape[1]
    gX = np.zeros((x1.shape[0],x2.shape[0],x1.shape[1]))

    code = \
    """
    for (int i=0; i<N; i++)
      for (int j=0; j<M; j++)
        for (int d=0; d<D; d++)
          gX(i,j,d) = (2/ls(d))*(x1(i,d) - x2(j,d));
    """
    # The C code weave above is 10x faster than this:
    # but scipy weave does not support python3
    for i in range(0, x1.shape[0]):
        gX[i,:,:] = 2 * (x1[i, :] - x2[:, :]) * (1 / ls)

    return gX


def se(ls, x1, x2=None, grad=False):
    ls = np.ones(ls.shape)
    d = dist2(ls, x1, x2)
    cov = np.exp(-0.5 * d)
    if grad:
        g = grad_ardse(ls, x1, x2)
        return (cov, g)
    else:
        return cov


def ardse(ls, x1, x2=None, grad=False, predictions=None):
    cov = np.exp(-0.5 * dist2(ls, x1, x2))
    if grad:
        return (cov, grad_ardse(ls, x1, x2))
    else:
        return cov


def grad_ardse(ls, x1, x2=None):
    r2 = dist2(ls, x1, x2)
    grad_r2 = grad_dist2(ls, x1, x2)
    return -0.5*np.exp(-0.5*r2)[:,:,np.newaxis] * grad_r2


def matern52(ls, x1, x2=None, grad=False):
    r2  = np.abs(dist2(ls, x1, x2))
    r   = np.sqrt(r2)
    cov = (1.0 + SQRT_5*r + (5.0/3.0)*r2) * np.exp(-SQRT_5*r)
    if grad:
        return (cov, grad_matern52(ls, x1, x2))
    else:
        return cov


def grad_matern52(ls, x1, x2=None):
    r       = np.sqrt(dist2(ls, x1, x2))
    grad_r2 = -(5.0/6.0)*np.exp(-SQRT_5*r)*(1 + SQRT_5*r)
    return grad_r2[:,:,np.newaxis] * grad_dist2(ls, x1, x2)
