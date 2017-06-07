# Copyright (C) 2015, Julien-Charles Lévesque <levesque.jc@gmail.com>
# Some bits taken from Jasper Snoek's Spearmint codebase:
# https://github.com/JasperSnoek/spearmint

import logging
import numpy as np

def init_search_space(config, **kwargs):
    return SearchSpace(config, **kwargs)


def get_variables_size(variables, one_hot_categorical=False):
    '''
    Get the size of the given configuration space and make sure it is valid
    '''
    size = 0
    # make sure each level only has one variable with the same name
    unique = {}
    for v in variables:
        if v['type'] not in ['int', 'float', 'enum']:
            raise Exception("Unknown parameter type.")

        if one_hot_categorical and v['type'] == 'enum':
            size += v['size'] * len(v['options'])
        else:
            size += v['size']

        if 'child_params' in v:
            size += get_variables_size(v['child_params'])

        if v['type'] == 'enum':
            v['options'] = np.array(v['options'])

        # deal with conflicts independently on each branch
        if 'parent' in v:
            unique[v['parent']] = unique.get(v['parent'], {})
            if v['name'] in unique[v['parent']]:
                raise Exception("Two variables with same name: %s." % v['name'])    
            unique[v['parent']][v['name']] = None
        elif v['name'] in unique:
            raise Exception("Two variables with same name: %s." % v['name'])
        unique[v['name']] = None
    return size


class SearchSpace:
    def __init__(self, variables, impute_inactive_values=False, 
            impute_value=0, rng=None,
            one_hot_categorical=True):
        # Count the total number of dimensions
        self.cardinality = get_variables_size(variables, one_hot_categorical)
        self.variables = variables

        self.impute_inactive_values = impute_inactive_values
        self.impute_value = impute_value

        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

        if self.impute_inactive_values and self.cardinality == len(variables):
            logging.warning("Imputation of inactive values is turned on,"
                " yet the provided search space seems to be flat.")

        self.one_hot_categorical = one_hot_categorical
        logging.info("Optimizing over %d dimensions\n" % (self.cardinality))

    def _index_map(self, u, items):
        u[u < 0] = 0
        u[u > 1] = 1
        out = np.floor((1 - np.finfo(float).eps) * u * float(items)).astype(int)
        return out

    def get_candidates(self, n):
        # Warning: this assumes everything is valid in the [0-1]^d unit cube
        X = self.rng.uniform(size=(n, self.cardinality))

        # do some fixing
        X = self.type_match(X)

        if self.impute_inactive_values:
            X = self.impute_inactive(X)

        return X

    def get_bounds(self):
        return [(0,1)] * self.cardinality

    def type_match(self, X, idx=-1, variables=None):
        '''
        Recursive function to sample and populate parameters with proper types.
        '''

        # root call will return only params
        root = False
        if idx == -1:
            root = True
            idx = 0
            variables = self.variables

        for v in variables:
            # manually fix one hot variables
            if v['type'] == 'enum' and self.one_hot_categorical:
                nc = len(v['options'])
                for dd in range(v['size']):
                    x = X[:, idx:idx+nc]
                    values = np.argmax(x, 1)
                    X[:, idx:idx+nc] = 0
                    X[:, idx:idx+nc][range(len(X)), values] = 1
                    idx += nc
            else:
                # increment idx, values returned are not important
                idx, value = self._get_var_value(X, idx, v)


            if 'child_params' in v:
                idx, params = self.type_match(X, idx,
                    v['child_params'])

        if root:
            return X
        else:
            return X, idx

    def get_dict_params(self, unitv, variables=None, idx=-1, params=None, parent=None):
        '''
        Goes from a single 0-1 bounded `u` vector to a dictionary containing 
        rescaled/mapped parameter values for the corresponding parameter names.
        '''
        if len(unitv.shape) == 2:
            if unitv.shape[0] == 1:
                unitv = unitv[0]
            else:
                # recursively call on bundled samples
                return [self.get_dict_params(u) for u in unitv]
            
        if unitv.shape[0] != self.cardinality:
            raise Exception("Hypercube dimensionality is incorrect.")

        unita = np.array([unitv])

        # root call will return only params
        root = False
        if idx == -1:
            root = True
            idx = 0
            params = {}
            variables = self.variables

        for v in variables:
            # fetch and increment idx
            idx, values = self._get_var_value(unita, idx, v)
            values = values[0] if len(values) == 1 else values

            active = v.get('parent', None) == parent
            if active:
                params[v['name']] = values

            if 'child_params' in v:
                unitv, idx, params = self.get_dict_params(unitv, 
                    v['child_params'], idx, params, parent=values)

        if root:
            return params
        else:
            return unitv, idx, params

    def impute_inactive(self, unitv, variables=None, idx=-1, parent=None):
        '''
        Does one pass over parameter space and imputes inactive parameters.
        '''
        unitv = unitv.copy()
        singlep = False
        if len(unitv.shape) == 1:
            unitv = np.array([unitv])
            singlep = True

        if unitv.shape[1] != self.cardinality:
            raise Exception("Hypercube dimensionality is incorrect.")

        root = False
        if idx == -1:
            root = True
            idx = 0
            variables = self.variables

        for v in variables:
            # fetch and increment idx
            prev_idx = idx
            idx, values = self._get_var_value(unitv, idx, v)

            if parent is not None:
                active = v.get('parent', None) == parent
                unitv[~active, prev_idx:idx] = self.impute_value

            if 'child_params' in v:
                unitv, idx = self.impute_inactive(unitv, 
                    v['child_params'], idx, parent=values[0])

        if singlep:
            unitv = unitv[0]

        if root:
            return unitv
        else:
            return unitv, idx

    def _get_var_value(self, unitv, idx, variable):
        values = []
        if variable['type'] == 'int':
            for dd in range(variable['size']):
                val = variable['min'] + self._index_map(unitv[:, idx],
                    variable['max'] - variable['min'] + 1)
                values.append(val)
                idx += 1
        elif variable['type'] == 'float':
            for dd in range(variable['size']):
                val = (variable['min'] + unitv[:, idx] *
                    (variable['max'] - variable['min']))
                # optional scale definition.
                scale = variable.get('scale', 'log')
                if scale == 'log':
                    val = 10**val
                values.append(val)
                idx += 1
        elif variable['type'] == 'enum':
            for dd in range(variable['size']):
                nc = len(variable['options'])
                if self.one_hot_categorical:
                    x = unitv[:, idx:idx+nc]
                    ii = np.argmax(x, 1)
                    idx += nc
                else:
                    ii = self._index_map(unitv[:, idx], nc)
                    idx += 1
                values.append(variable['options'][ii])
        else:
            raise Exception("Unknown parameter type.")
        return idx, np.array(values).flatten()
