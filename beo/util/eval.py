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
import multiprocessing
import signal
import time
import traceback

import numpy as np
from sklearn.metrics import accuracy_score
import psutil


class ExpTimeout(Exception):
    pass


def alarm_handler(signum, frame):
    raise ExpTimeout("Out of time.")


def error_score(y_true, y_pred):
    return 1 - accuracy_score(y_true, y_pred)


def kill_proc_tree(pid, including_parent=True):
    # util function to kill a process and its children
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.terminate()
    gone, alive = psutil.wait_procs(children, timeout=1)
    for p in alive:
        p.kill()
    if including_parent:
        parent.kill()


def train_test_learner(learner_class, learner_fixed_params, job_params,
        datasets,
        retrain=True,
        distinct_process=True, 
        crash_on_exception=True,
        max_eval_time=None,
        invalid_job_value=1,
        return_predictions=True):
    '''
    Runs a single experiment specified in `job_params` and saves the result
     and hyper-parameters in intermediate files val_results.csv and
     hyper_params.csv within the folder `exp_params['output_folder']`.

    Parameters:
    -----------
    learner: One or more learner classes
    learner_fixed_params: Parameters which don't change for the learners

    job_params: Parameters concerning the current job which will be given
        to the learners.

    datasets: dictionary containing `train_data`, `val_data`, `test_data`,
        and corresponding `_labels

    invalid_job_value: Value to set for a job that was killed (because it was
         too long or because it crashed). Should be something higher than the
         worst possible value for the current setting.

    retrain: if true, retrain model on all available validation data

    return_predictions: if true, will save model predictions and return them
    '''
    time_start = time.time()
    learner_params = dict(learner_fixed_params)

    if 'test_data' not in datasets:
        logging.warning("No test split given, test predictions will not be"
            " computed!")

    #Regular hyperparameters structure
    for n, p in job_params.items():
        #Parameters are wrapped in arrays for spearmint
        #quick and dirty fix because all the functions are used to
        #receive non iterables
        try:
            if len(p) == 1:
                cur_p = p[0]
            else:
                cur_p = p
        except:
            # Un-iterable
            cur_p = p
        learner_params[n] = cur_p

    def timed_func(out_queue=None):
        try:
            logging.info("Running train_test_learner subfunc with out_queue=%s"
                % out_queue)
            results = {}

            if return_predictions:
                results['train_preds'] = []
                results['val_preds'] = []
                results['test_preds'] = []

            fresults = []
            for X_train, y_train, X_val, y_val in zip(
                    datasets['train_data'], datasets['train_labels'], 
                    datasets['val_data'], datasets['val_labels']):
                res_i = {}
                learner = learner_class(**learner_params)
                #train model
                learner.fit(X_train, y_train)

                train_preds = learner.predict(X_train)
                val_preds = learner.predict(X_val)

                res_i['train_T_err'] = error_score(y_train, train_preds)
                res_i['val_T_err'] = error_score(y_val, val_preds)

                #compute accuracy on testing split
                if 'test_data' in datasets:
                    test_preds = learner.predict(datasets['test_data'])
                    res_i['test_T_err'] = error_score(datasets['test_labels'], 
                        test_preds)

                fresults.append(res_i)

                if return_predictions:
                    # we never really need training predictions, so skip em
                    results['val_preds'].append(val_preds)
                    results['test_preds'].append(test_preds)

            # store everything
            results['cross_val_folds'] = fresults
            
            #average error rates measured on the k folds
            results['train_T_err'] = np.average([r['train_T_err'] for r in fresults])
            results['train_T_std'] = np.std([r['train_T_err'] for r in fresults])
            results['val_T_err'] = np.average([r['val_T_err'] for r in fresults])
            results['val_T_std'] = np.std([r['val_T_err'] for r in fresults])

            if 'test_data' in datasets:
                results['test_T_err'] = np.average([r['test_T_err'] for r in fresults])
                results['test_T_std'] = np.std([r['test_T_err'] for r in fresults])
    
                #if there was a test data provided, retrain the model on the
                # whole train + valid data
                if 'tv_data' in datasets and retrain:
                    learner = learner_class(**learner_params)
                    learner.fit(datasets['tv_data'], datasets['tv_labels'])
                    test_tv_preds = learner.predict(datasets['test_data'])
                    results['test_TV_err'] = error_score(
                        datasets['test_labels'], test_tv_preds)

                    if return_predictions:
                        results['test_tv_preds'] = test_tv_preds

            results['raised_exception'] = False
        except ExpTimeout as exc:
            # This exception should only be raised when sigalarm is set,
            # let main level handle this ran out of time exception.
            raise
        except Exception as exc:
            results['raised_exception'] = True
            results['exception'] = traceback.format_exc()
            logging.info(results['exception'])

        if out_queue is not None:
            logging.debug("train_test_learner.timed_func: putting results"
                " back in queue.")
            out_queue.put(results)
            logging.debug("train_test_learner.timed_func: returning.")
            return
        else:
            return results

    results = {}
    if distinct_process:
        # there must not be any unpicklable objects in the parameters given to
        # timed_func
        queue = multiprocessing.Queue()
        proc = multiprocessing.Process(target=timed_func, args=(queue,))
        logging.info("Starting learning process.")
        proc.start()
        logging.info("Calling join with timeout %s" % max_eval_time)
        proc.join(timeout=max_eval_time)
        logging.info("Join finished, process exitcode: %s" % proc.exitcode)

        if proc.is_alive():
            timed_out = True
            logging.info("Timed out, killing process tree, including children.")
            kill_proc_tree(proc.pid)
        elif proc.exitcode != 0:
            timed_out = False
            results = {'raised_exception': True,
                'exception': "Job exited with code %i" % proc.exitcode}
        else:
            timed_out = False
            try:
                results = queue.get(timeout=1)
            except multiprocessing.queues.Empty as exc:
                logging.error("Queue was empty, process crashed somewhere?")
                results = {'raised_exception': True,
                    'exception': "Job exited with code %i" % proc.exitcode}
    else:
        logging.warning("Do NOT use SIGALRM with Theano because by default"
            " it'll intercept the signal and not raise it. Try to use"
            " the distinct_process option instead.")
        #Setup timeout machinery, won't work on non-unix systems
        signal.signal(signal.SIGALRM, alarm_handler)
        #very important to have this function call bundled in a subprocess,
        # otherwise the signal can interrupt other stuff
        max_eval_time = 0 if max_eval_time is None else max_eval_time
        signal.alarm(int(max_eval_time))

        try:
            logging.info("Calling learning with sigalrm set for timer %s"
                % (max_eval_time))
            results = timed_func()
            logging.info("Learning completed within time limit.")
            timed_out = False
            signal.alarm(0)
        except ExpTimeout as exc:
            timed_out = True
            signal.alarm(0)
        except:
            raise

    raised_exception = results.get('raised_exception', False)
    if not timed_out and not raised_exception:
        #Results pre-combined
        val = results['val_T_err']
        results['timeout'] = False
        results['raised_exception'] = False
        val_err = val
    else:
        if not crash_on_exception and raised_exception:
            logging.info("Learning raised an exception: %s"
               % results['exception'])
        elif raised_exception:
            # Give out some information before crashing.
            logging.info("Learning raised an exception, ending experiment."
                " Params: ", job_params)
            raise Exception(results['exception'])
        else:
            logging.info("Learning taking too long (limit: %s), killed it."
               % max_eval_time)

        results = {}
        #set invalid value to make it easier to filter
        results['val_T_err'] = invalid_job_value
        results['test_T_err'] = invalid_job_value
        results['test_TV_err'] = invalid_job_value
        results['timeout'] = timed_out
        results['raised_exception'] = raised_exception
        val_err = invalid_job_value

    results['train_test_time'] = time.time() - time_start

    #Get the error rather than accuracy and scale it so the noise std becomes
    return val_err, results
