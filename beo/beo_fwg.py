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

import logging

import numpy as np

from beo import bo
from beo.util import trace
from beo import ensemble


def get_valid_jobs(expt, expt_trace):
    # Locate valid jobs (with completed predictions) from the expt trace
    # TODO: maybe do it directly from the prediction cache?
    if len(expt_trace) > 0:
        T = trace.inside_out(expt_trace, filter_keys=['choices'], 
            verbose=False)
        valid_jobs = (~T['raised_exception']) & (~T['timeout'])
        valid_job_ids = T['job_id'][valid_jobs]

        invalid_jobs = T['raised_exception'] | T['timeout']
        invalid_job_ids = T['job_id'][invalid_jobs]
    else:
        valid_jobs = []
        valid_job_ids = []

        invalid_jobs = []
        invalid_job_ids = []
    return valid_jobs, valid_job_ids, invalid_jobs, invalid_job_ids


class BeoFwg(bo.BayesianOptimizer):
    '''
    Utility object to perform a hyper-parameter optimization. Provides
     fitness values for a forward greedy ensemble construction procedure.
    '''
    def __init__(self, ensemble_loss_func,
            replace_members, n_classifiers, 
            **kwargs):
        '''
        Parameters:
        -----------

        ensemble_loss_func: function determining the `loss` of an ensemble given
            its predictions

        replace_members: will leave classifiers in the pool instead of removing
            them, effectively allowing the same classifier to be present 
            multiple times in the ensemble

        n_classifiers: number of classifiers in the ensemble
        '''

        # Force the base class to store predictions on validation split(s)
        kwargs['store_pred_cache'] = True
        super().__init__(**kwargs)

        self.num_classifiers = n_classifiers
        self.ensemble_loss_func = ensemble_loss_func
        self.replace_members = replace_members

        # assume all labels are present in the test set
        self.u_labels = np.unique(self.test_pred_cache['labels'][0])

    # Evaluate the performance of a potential ensemble at given points
    def eval(self, expt, expt_trace, complete, values):
        return_info = {}

        bo.sync_hdf5_pred_cache(self.pred_cache_file, self.val_pred_cache)

        #Whose turn is it in the 'round-robin'?
        self.i = expt.get('i', 0) % self.num_classifiers
        logging.info("Optimizing classifier %i" % self.i)

        #If the last iteration wasn't on the same classifier,
        #reset the chooser module to force a restart and burn-in
        if len(expt_trace) > 0 and expt_trace[-1]['i'] != self.i:
            return_info['clear_chooser_pkl'] = True

        losses_with_failed, ens_losses, ids, cur_ensemble = \
            self.compute_loss_refresh_member_i(self.i,
            expt, expt_trace, values)

        # increment for next iteration, expt is saved on file system and
        #  distributed to other nodes
        expt.i = self.i + 1

        return losses_with_failed, ids, return_info

    def update(self, job_id, job_results, expt, expt_trace,
            complete, values):
        # Set during first call to eval, the index of the classifier
        # targeted by this iteration. since expt.i can be affected by other
        # processes running fwgo, we need to store it explicitly
        i = self.i
        losses_with_failed, ens_losses, ids, cur_ensemble = \
            self.compute_loss_refresh_member_i(i, expt,
            expt_trace, values, update_ensemble=True)

        if len(ens_losses) > 0:
            #compute test predictions for given ensemble --
            cur_ensemble_ids = list(cur_ensemble['choices'].values())

            # Test predictions are always evaluated the same, they're just
            # not exactly computed on the same data.
            ens_test_error, ens_test_loss = ensemble.eval_ensemble(
                cur_ensemble_ids, self.test_pred_cache, 
                self.ensemble_loss_func, regression=False, 
                u_labels=self.u_labels)

            # Give the proper label to testing error, as they are not the same
            if self.eval_params.get('retrain', True):
                cur_ensemble['test_TV_err'] = ens_test_error
            else:
                cur_ensemble['test_T_err'] = ens_test_error

            #losses = np.average(ensemble_loss_func(val_labels, predictions))

        # return dict that will be stored in the trace
        return {'ensemble': cur_ensemble, 'i': i}

    def compute_loss_refresh_member_i(self, i, expt, expt_trace, values,
            update_ensemble=True, bag_classifiers=False):
        '''
        Updates member `i` in the ensemble, i.e. remove it and compute losses
         for putting back any other classifier in the pool
        '''
        #current ensemble, stored in the resilient expt structure
        cur_ensemble = expt.get('cur_ensemble', {'choices': {}})

        choices = cur_ensemble['choices'].copy()
        #remove classifier for current iteration (i.e. make a choice again)
        if i in choices:
            choices.pop(i)

        valid_jobs, valid_job_ids, invalid_jobs, invalid_job_ids =\
            get_valid_jobs(expt, expt_trace)

        if self.replace_members:
            # leave chosen classifiers in the pool
            remaining_complete = valid_job_ids
        else:
            remaining_complete = np.array([idx for idx in valid_job_ids if
                idx not in choices.values()], dtype=int)

        if bag_classifiers and len(remaining_complete) > 0:
            remaining_complete = np.random.choice(remaining_complete, 
                int(0.66*len(remaining_complete)), replace=False)

        #measure fitnesses    
        logging.info("Choices before eval: %s" % choices)
        ens_errors, ens_losses = ensemble.eval_combinations(
            list(choices.values()), remaining_complete, self.val_pred_cache, 
            self.ensemble_loss_func, False, self.u_labels)

        #Make a new choice given the pool of classifiers available
        if len(ens_errors) > 0:
            # Sorts by last key first, ens_losses, then individual validation
            # errors, with a last randomly sampled tiebreaking entry
            min_i = np.lexsort((
                np.random.sample(len(ens_losses)), 
                values[remaining_complete], 
                ens_losses))[0]
            my_choice = remaining_complete[min_i]

            if update_ensemble:
                cur_ensemble['choices'][i] = my_choice
                cur_ensemble['val_T_err'] = ens_errors[min_i]
                expt.cur_ensemble = cur_ensemble    
                logging.info("Choices: %s" % cur_ensemble['choices'])
                logging.info("Single best validation error: %f" % 
                    np.min(values))
                logging.info("Ensemble validation error : %f" % 
                    ens_errors[min_i])

        logging.debug("Ensemble losses (%s): %s" % (self.ensemble_loss_func,
            ens_losses))
        
        invalid_jobs_losses = values[invalid_job_ids]

        # Put back losses of incomplete jobs
        losses_with_failed = np.hstack((ens_losses, invalid_jobs_losses))
        ids = np.hstack((remaining_complete, invalid_job_ids)).astype(int)

        return losses_with_failed, ens_losses, ids, cur_ensemble
