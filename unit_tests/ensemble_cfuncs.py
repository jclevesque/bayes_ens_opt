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

"""
This file was created for internal testing/debugging of the ensemble c functions
"""
import unittest
import numpy as np

import beo.ensemble_cfuncs as ens_c
# private external dependency
import learning_toolbox.ensemble as ens


class EnsembleFuncsTest(unittest.TestCase):
    def test_majority_vote(self):
        n = 1000
        m = 100
        nb_classes = 10
        votes = np.random.randint(0, nb_classes, (n, m))
        # labels = np.random.randint(0, nb_classes, n)

        c_pred = ens_c.majority_vote(votes)
        ltb_pred = ens.py_majority_vote(votes)

        self.assertTrue(np.all(ltb_pred == c_pred))

    def test_w_vote(self):
        n = 1000
        m = 100
        nb_classes = 10
        votes = np.random.randint(0, nb_classes, (n, m))
        # labels = np.random.randint(0, nb_classes, n)

        w = np.random.sample(100)
        w = w / np.sum(w)

        c_pred = ens_c.weighted_vote(votes, w)
        ltb_pred = ens.py_weighted_vote(votes, w)

        self.assertTrue(np.all(ltb_pred == c_pred))

if __name__ == '__main__':
    unittest.main()
