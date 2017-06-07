import numpy as np


def argmax_ties(X):
    random = np.random.sample(len(X))
    # Last column is primary sort key!
    indices = np.lexsort((random.flatten(), X.flatten()))
    return indices[-1]


def argmin_ties(X):
    return argmax_ties(-X)


# Copyright (C) 2012 Jasper Snoek, Hugo Larochelle and Ryan P. Adams
# https://github.com/JasperSnoek/spearmint/blob/master/spearmint/spearmint/util.py
def slice_sample(init_x, logprob, sigma=1.0, step_out=True, max_steps_out=1000,
                 compwise=False, verbose=False, rng=None):
    if rng is None:
        rng = np.random.RandomState()

    def direction_slice(direction, init_x):
        def dir_logprob(z):
            return logprob(direction*z + init_x)

        upper = sigma*rng.rand()
        lower = upper - sigma
        llh_s = np.log(rng.rand()) + dir_logprob(0.0)

        l_steps_out = 0
        u_steps_out = 0
        if step_out:
            while dir_logprob(lower) > llh_s and l_steps_out < max_steps_out:
                l_steps_out += 1
                lower       -= sigma
            while dir_logprob(upper) > llh_s and u_steps_out < max_steps_out:
                u_steps_out += 1
                upper       += sigma

        steps_in = 0
        while True:
            steps_in += 1
            new_z     = (upper - lower)*rng.rand() + lower
            new_llh   = dir_logprob(new_z)
            if np.isnan(new_llh):
                raise Exception("Slice sampler got a NaN")
            if new_llh > llh_s:
                break
            elif new_z < 0:
                lower = new_z
            elif new_z > 0:
                upper = new_z
            else:
                raise Exception("Slice sampler shrank to zero!")

        if verbose:
            print("Steps Out:", l_steps_out, u_steps_out, " Steps In:", 
                steps_in)

        return new_z*direction + init_x

    if not init_x.shape:
        init_x = np.array([init_x])

    dims = init_x.shape[0]
    if compwise:
        ordering = list(range(dims))
        rng.shuffle(ordering)
        cur_x = init_x.copy()
        for d in ordering:
            direction    = np.zeros((dims))
            direction[d] = 1.0
            cur_x = direction_slice(direction, cur_x)
        return cur_x

    else:
        direction = rng.randn(dims)
        direction = direction / np.sqrt(np.sum(direction**2))
        return direction_slice(direction, init_x)
