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
from sklearn.metrics import r2_score, accuracy_score

from beo.ensemble_cfuncs import majority_vote


def eval_combinations(choices, remaining_complete, pred_cache,
        ensemble_loss_func, regression, u_labels):
    '''
    Given a pool of classifiers described by validation predictions and their
        corresponding job ids in `val_pred_cache`, and a fixed subset
        of classifiers selected from this pool by other processes `choices`,
        evaluate the performance of a new ensemble with any of the remaining
        classifiers.

    For example, say we have c1, c2 fixed, and we must choose one from
        c3, c4. We will evaluate the performance of ensembles (c1, c2, c3),
        and (c1, c2, c4).
    '''
    errs = np.zeros(len(remaining_complete))
    ens_losses = np.zeros(len(remaining_complete))

    for i, idx in enumerate(remaining_complete):
        cur_ensemble_ids = list(choices) + [idx]
        errs[i], ens_losses[i] = eval_ensemble(cur_ensemble_ids, pred_cache,
            ensemble_loss_func, regression, u_labels)

    return errs, ens_losses


def eval_ensemble(choices, pred_cache, ensemble_loss_func, 
        regression, u_labels, testing=False):
    '''
    Evaluate the performance of an ensemble, returns both the empirical error
     rate (zero-one loss) and loss function (given by `ensemble_loss_func`)
    '''
    labels = pred_cache['labels']

    # infer number of validation folds from the predictions
    k = len(pred_cache[choices[0]])

    k_errs = []
    k_losses = []
    for i in range(k):
        ind_predictions = np.array([pred_cache[jid][i] for jid in
            choices]).T
        if regression:
            ens_preds = np.average(ind_predictions, 1)
            err_i = -r2_score(labels[i], ens_preds)
            loss_i = np.average(ensemble_loss_func(labels[i], ind_predictions,
                ens_preds, u_labels))
        else:
            #for now use majority vote
            ens_preds = majority_vote(ind_predictions)
            err_i = 1 - accuracy_score(labels[i], ens_preds)
            loss_i = np.average(ensemble_loss_func(labels[i], ind_predictions,
                ens_preds, u_labels))
        k_errs.append(err_i)
        k_losses.append(loss_i)
    err = np.average(k_errs)
    loss = np.average(k_losses)

    return err, loss
