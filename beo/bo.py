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

import bz2
import importlib
import logging
import os
import pickle
import time

import h5py

from beo.util.eval import train_test_learner
from beo.util import file_lock
from beo.util import file_handling


class BayesianOptimizer:
    def __init__(self, search_space, output_folder,
            chooser_mod, 
            expt_mgr_mod,
            eval_params,
            func_scale=1,
            chooser_params={},
            expt_mgr_params={},
            max_iters=100, n_candidates=10000,
            pre_processing=None, post_processing=None,
            log_name='', log_level=logging.INFO,
            store_pred_cache=True):
        '''
        Parameters:
        -----------

        search_config (dict): the search space to optimize, will
            directly be unpacked as arguments to the Experiment module

            TODO: Experiment should be static, SearchSpace should be defined by
            the search_config param

        output_folder (string): folder in which to dump all outputs, both temporary
            and permanent files

        chooser: name of the module to import from
            learning_toolbox.smbo.choosers, which will effectively choose the next
            hyperparameters to evaluate at each iteration

        func_scale: multiply the observed values of the function (of type
            `err_type`) by this factor to help with the chooser module.
            For example, GPEIOptChooser uses a prior of 1 on the standard
            deviation of observed function values.

        n_candidates: number of candidates to sample from the search space

        max_iters:
        pre_processing
        post_processing
        log_name
        log_level
        '''
        if log_name != '':
            logging.root.name = log_name  # this essentially sets the name of the root logger
        logging.root.setLevel(log_level)
        self.expt_cls = importlib.import_module(expt_mgr_mod).Experiment

        #Setup output folder.
        self.expt_dir = output_folder
        os.makedirs(output_folder, exist_ok=True)

        #Do the pre processing funcs if any
        if pre_processing is not None and \
                not os.path.exists(os.path.join(self.expt_dir, 'pre_processing_done')):
            logging.info("Executing pre-processing funcs: %s."
                % pre_processing)
            pp_lock = file_lock.Locker(os.path.join(self.expt_dir, 'pre_processing'))
            try:
                pp_lock.acquire(timeout=1)

                #At this point the lock is acquired.
                for pp_func in pre_processing:
                    pp_func()

                f = open(os.path.join(self.expt_dir, 'pre_processing_done'), 'wt')
                f.close()

                pp_lock.release()
            except file_lock.LockTimeout as exc:
                raise Exception("Pre processing already being executed on"
                    " another node?")

        # Grabbed in this function params, need to be forwarded to next level
        self.eval_params = eval_params
        self.expt_trace_fn = os.path.join(self.expt_dir, 'expt_trace.pkl.bz2')

        # Save parameters
        self.search_space = search_space

        self.chooser_mod = chooser_mod
        self.chooser_params = chooser_params

        self.expt_mgr_mod = expt_mgr_mod
        self.expt_mgr_params = expt_mgr_params

        self.func_scale = func_scale
        self.max_iters = max_iters
        self.n_candidates = n_candidates
        self.pre_processing = pre_processing
        self.post_processing = post_processing
        self.store_pred_cache = store_pred_cache

        if self.store_pred_cache:
            self.pred_cache_file = self.expt_dir + '/pred_cache.hdf5'
            if not os.path.exists(self.pred_cache_file):
                f = h5py.File(self.pred_cache_file, 'w', libver='latest')
                f.require_group('predictions').attrs['dtype'] = 'int'
                f.close()
            self.init_caches()
        else:
            self.pred_cache_file = None

    def init_caches(self):
        # keys are job_ids, values are predictions themselves
        # exception for 'labels' containing the real labels
        self.val_pred_cache = {'labels': 
            self.eval_params['datasets']['val_labels']}

        # small hack since test error is computed using the same function as
        # training error etc.
        self.test_pred_cache = {'labels': 
            [self.eval_params['datasets']['test_labels']]}

    def optimize(self):
        ############################
        # main loop
        #############################
        jobs_running = []

        if os.path.exists(self.expt_trace_fn):
            expt_trace = pickle.load(bz2.open(self.expt_trace_fn, 'rb'))
            it = len(expt_trace)
        else:
            it = 0
        while it < self.max_iters:
            iter_start = time.time()

            #capture lock on grid
            job_params, job_id, info_from_train = self.get_next_job()
            #release lock on grid

            logging.debug("Chose job_params %s" % job_params)

            chooser_time = time.time() - iter_start

            # Jobs that take too long to compute will be assigned this value
            # same goes for jobs that result in an error if the 
            # eval_params['crash_on_exception'] parameter was set to false. 
            # In terms of validation error, this is like saying these have
            # 100% error rate.
            invalid_job_val = 1
            if self.func_scale < 0:
                # if we are trying to maximize, need a different value for
                # invalid jobs
                invalid_job_val = -1

            # Train and test with the given hyperparameters
            job_err, job_stats = train_test_learner(
                job_params=job_params,
                invalid_job_value=invalid_job_val,
                **self.eval_params
                )

            job_stats['chooser_time'] = chooser_time
            job_stats['iteration_time'] = time.time() - iter_start

            #save its results
            it, jobs_running = self.update_job_results(job_stats, job_id,
                job_params, job_err)

        #Do the post processing funcs if any
        if len(jobs_running) > 0:
            logging.info("Jobs (%s) still running on another node/proc." % jobs_running)
        elif self.post_processing is not None:
            self.exec_post_processing()
            f = open(self.expt_dir + '/post_proc_done', 'wt')
            f.close()

    def get_next_job(self):
        #######
        # After this point, the expt grid is locked and we have exclusive
        # access to the experiment files and folders
        #######
        with self.expt_cls(self.expt_dir, self.search_space, init=True,
                **self.expt_mgr_params) as expt:

            X, values, pending_i, complete_i = expt.get_all()

            expt_trace_fn = os.path.join(self.expt_dir, 'expt_trace.pkl.bz2')
            if os.path.exists(expt_trace_fn):
                expt_trace = pickle.load(bz2.open(expt_trace_fn, 'rb'))
            else:
                expt_trace = []

            # Update the prediction caches if store_pred_cache is true
            self.sync_caches()

            #Update ensemble and get the ids of completed jobs which haven't
            # been picked so far
            losses, remaining_complete_i, info_from_train = self.eval(
                expt, expt_trace, complete_i, values)

            ###################
            # Get the next job

            # Initialize chooser module
            chooser_module = importlib.import_module(self.chooser_mod)

            #If the last iteration wasn't completed by us, reinitialize the chooser
            # to fresh hyper-hyperparams
            if info_from_train.get('clear_chooser_pkl', False):
                logging.debug("Fresh chooser restart.")
                chooser_pkl = self.expt_dir + '/' + chooser_module.__name__ + '.pkl'
                if os.path.exists(chooser_pkl):
                    os.remove(chooser_pkl)

            chooser = chooser_module.init(self.expt_dir, expt.search_space, 
                self.chooser_params)

            # Warning: there is no filtering of completed candidates for
            # grid-based search spaces (be it sobol, axis-aligned grid)
            candidate = self.search_space.get_candidates(
                self.n_candidates)

            n_candidates = candidate.shape[0]
            n_pending = pending_i.shape[0]
            n_complete = complete_i.shape[0]
            logging.info("%d candidates   %d pending   %d complete" %
                (n_candidates, n_pending, n_complete))

            next_job_x = chooser.next(losses * self.func_scale,
                candidate, X[pending_i], X[remaining_complete_i])

            # Add the new candidate to the Experiment
            next_job_id = expt.add_job(next_job_x)
            next_job_params = self.search_space.get_dict_params(next_job_x)

            # Run everything in the same process. The actual job will be executed
            # once this function returns
            expt.set_submitted(next_job_id, os.getpid())
            expt.set_running(next_job_id)

        return next_job_params, next_job_id, info_from_train

    def update_job_results(self, job_stats, job_id, job_params, job_err):
        expt_trace_fn = os.path.join(self.expt_dir, 'expt_trace.pkl.bz2')

        #Lock and fetch the experiment grid
        with self.expt_cls(self.expt_dir, self.search_space) as expt:
            expt.set_complete(job_id, job_err,
                job_stats['train_test_time'], job_stats['timeout'])
            X, values, pending_i, complete_i = expt.get_all()

            if os.path.exists(expt_trace_fn):
                expt_trace = pickle.load(bz2.open(expt_trace_fn, 'rb'))
            else:
                expt_trace = []

            self.sync_caches(job_id, job_stats)

            # Update procedure optimization, redefine `update` method
            extra_stats = self.update(job_id, job_stats, expt,
                expt_trace, complete_i, values)

            job_stats['job_id'] = job_id
            job_stats['job_params'] = job_params
            job_stats['job_x'] = X[job_id]
            job_stats.update(extra_stats)
            expt_trace.append(job_stats)

            file_handling.safe_save_pickle(expt_trace, expt_trace_fn)

            n_complete = len(complete_i)
            pending = expt.get_pending()

        return n_complete, pending

    def eval(self, expt, expt_trace, complete_i, values):
        # Default evaluation is just the empirical error, don't need
        # to construct an ensemble
        return values[complete_i], complete_i, {}

    def update(self, job_id, job_stats, expt,
            expt_trace, complete_i, values):
        # Nothing to update, nothing to put in the stats dict
        return {}

    def sync_caches(self, job_id=None, job_stats=None):
        # Skip this step if caches are not required
        if not self.store_pred_cache:
            return

        # Update the prediction caches if given a completed job
        if job_id is not None and not (job_stats['timeout'] or
                job_stats['raised_exception']):
            update_hdf5_pred_cache(self.pred_cache_file, 
                job_id, job_stats)

        sync_hdf5_pred_cache(self.pred_cache_file, self.val_pred_cache)
        sync_hdf5_pred_cache(self.pred_cache_file, 
            test_cache=self.test_pred_cache)

    def exec_post_processing(self):
        logging.info("Executing %i post processing funcs."
            % len(self.post_processing))
        pp_lock = file_lock.Locker(os.path.join(self.expt_dir, 'post_processing'))
        try:
            pp_lock.acquire(timeout=1)
            #At this point the lock is acquired.
            for pp_func in self.post_processing:
                pp_func()
            pp_lock.release()
        except file_lock.LockTimeout:
            logging.info("Post processing already being executed on"
                " another node?")


def update_hdf5_pred_cache(cache_file, completed_job_id, completed_job_stats):
    '''
    Put the predictions of completed job `completed_job_id` in an
    hdf5 cache.
    '''

    # Setup hdf5 file
    f = h5py.File(cache_file, 'a', libver='latest')
    dtype = f['predictions'].attrs['dtype']

    #Update the cache of validation predictions
    val_group = f.require_group('/predictions/%i/val' % (completed_job_id))
    fill_hdf5_split_preds(val_group, completed_job_stats['val_preds'], 
        dtype)

    test_group = f.require_group('/predictions/%i/test' % (completed_job_id))
    fill_hdf5_split_preds(test_group, [completed_job_stats['test_tv_preds']], 
        dtype)
    f.close()


def fill_hdf5_split_preds(hdf5_group, preds, dtype):
    for i in range(len(preds)):
        hdf5_group.create_dataset('%i' % i, data=preds[i], compression='gzip')
    return True


def sync_hdf5_pred_cache(cache_file, val_cache=None, test_cache=None):
    f = h5py.File(cache_file, 'r', libver='latest')
    if val_cache is not None:
        ids = list(f['predictions'].keys())
        for job_id in ids:
            entry = 'predictions/%s/val' % job_id
            if job_id not in val_cache and entry in f:
                reps = sorted(list(f[entry]))
                val_cache[int(job_id)] = [f['predictions/%s/val/%s' %
                    (job_id, r)][:] for r in reps]

    if test_cache is not None:
        ids = list(f['predictions'].keys())
        for job_id in ids:
            entry = 'predictions/%s/test' % job_id
            if job_id not in test_cache and entry in f:
                reps = sorted(list(f[entry]))
                test_cache[int(job_id)] = [f['predictions/%s/test/%s' 
                    % (job_id, r)][:] for r in reps]
    f.close()


def reset_job_status(expt_dir, job_id, expt_cls, search_space):
    with expt_cls(expt_dir, search_space) as expt:
        expt.set_candidate(job_id)
