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

import numpy as np


def inside_out(results, filter_keys=[], verbose=False):
    '''
    Flips the results array inside out. At first, when output by grid
    search and array grid search, the array is a list of dictionaries
    (sometimes containing more dictionaries). Make this into a dict of
    lists, allowing for easier browsing of the results.

    Parameters:
    -----------

    results: list of dict, as constructed by the function test.grid_search
    (and then reloaded from the files saved)

    Returns:
    --------

    dict_of_lists: a dictionary containing a list of items for each of the keys
     found

    Example:
    --------

    >>> a = [{'x':8,'y':0},{'x':15,'y':20}]
    >>> results_array.inside_out(a)
    {'x': array([ 8, 15]), 'y': array([ 0, 20])}
    '''

    # Variant implementation
    dict_of_lists = {}
    if len(results) == 0:
        return dict_of_lists

    # Grab all possible keys
    keys = np.unique(np.hstack([list(r.keys()) for r in results]))
    # First pass: flatten the first level
    for key in keys:
        if isinstance(key, str) and key in filter_keys:
            continue
        try:
            dict_of_lists[key] = np.array([d[key] for d in results])
        except KeyError as exc:
            #one of the guys didnt have the key, uneven results structure...
            #let it pass by
            if verbose:
                print("Missing key from one of the jobs, ignoring", exc)

    # Second pass: recursively deal with embedded dictionaries
    for key, val in dict_of_lists.items():
        if isinstance(val[0], dict):
            dict_of_lists[key] = inside_out(val, filter_keys, verbose)

    return dict_of_lists
