# Copyright (C) 2015, Julien-Charles Lévesque <levesque.jc@gmail.com>
# Some bits taken from Jasper Snoek's Spearmint codebase:
# https://github.com/JasperSnoek/spearmint

import bz2
import logging
import os
import pickle

import numpy as np

import beo.util.file_lock as lock
import beo.util.file_handling as ufile

CANDIDATE_STATE = 0
SUBMITTED_STATE = 1
RUNNING_STATE = 2
COMPLETE_STATE = 3
BROKEN_STATE = -1

# name of the file in which to save the expt grid
EXPERIMENT_FILE = 'expt.pkl.bz2'

# do not pickle these class variables
SKIP_SAVE_LOAD = ['locker', 'pkl']
SKIP_LOAD = ['search_space']

class Job:
    def __init__(self, x, status=CANDIDATE_STATE, duration=None,
            timed_out=None, proc_id=None):
        self.x = x
        self.status = status
        self.duration = duration
        self.timed_out = timed_out
        self.proc_id = proc_id
        self.result = None


class BaseExperiment:
    def __init__(self, expt_dir, search_space, init=False):
        self.expt_dir = expt_dir
        self.search_space = search_space
        self.pkl = os.path.join(expt_dir, EXPERIMENT_FILE)
        self.locker = lock.Locker(self.pkl)

        os.makedirs(expt_dir, exist_ok=True)

        # Only one process at a time is allowed to have access to this object
        self.locker.acquire()

        # First run, 
        if init and not os.path.exists(self.pkl):
            self.jobs = []
        # Or load from the pickled file.
        else:
            self._load()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._save()
        self.locker.release()

    def get(self, attr, default):
        ''' Dict-like get function, make things easier'''
        if hasattr(self, attr):
            return self.__getattribute__(attr)
        else:
            return default

    def get_jobs_status(self):
        return np.array([j.status for j in self.jobs], dtype=int)

    def get_pending(self):
        status = self.get_jobs_status()
        return np.nonzero((status == SUBMITTED_STATE) |
                          (status == RUNNING_STATE))[0]

    def get_complete(self):
        return np.nonzero(self.get_jobs_status() == COMPLETE_STATE)[0]

    def get_x(self):
        return np.array([j.x for j in self.jobs])

    def get_all(self):
        pend = self.get_pending()
        comp = self.get_complete()
        vals = self.get_values()
        X = self.get_x()
        return X, vals, pend, comp

    def get_values(self):
        return np.array([j.result for j in self.jobs], dtype=float)

    def get_complete_not_timed_out(self):
        return np.array([i for i, job in enumerate(self.jobs)
                if (job.status == COMPLETE_STATE and not job.timed_out)])

    def get_broken(self):
        return np.nonzero(self.get_jobs_status() == BROKEN_STATE)[0]

    def get_best(self):
        status = self.get_jobs_status()
        complete = np.nonzero(status == COMPLETE_STATE)[0]
        values = [self.jobs[i].value for i in complete]
        if len(complete) > 0:
            idx = np.argmin(values)
            cur_min = values[idx]
            return cur_min, idx
        else:
            return np.nan, -1

    def get_proc_id(self, jid):
        return self.jobs[jid].proc_id

    def add_job(self, x):
        job = Job(x)
        self.jobs.append(job)

        return len(self.jobs) - 1

    def set_candidate(self, jid):
        logging.info("Remove job %i from job list." % jid)
        self.jobs.pop(jid)

    def set_submitted(self, jid, proc_id):
        self.jobs[jid].status = SUBMITTED_STATE
        self.jobs[jid].proc_ids = proc_id

    def set_running(self, jid):
        self.jobs[jid].status = RUNNING_STATE

    def set_complete(self, jid, result, duration, timed_out=False):
        job = self.jobs[jid]
        job.status = COMPLETE_STATE
        job.result = result
        job.duration = duration
        job.timed_out = timed_out
        #job.error =

    def set_broken(self, jid):
        self.jobs[jid].status = BROKEN_STATE

    def _load(self):
        fh = bz2.BZ2File(self.pkl, 'rb')
        saved_dict = pickle.load(fh)
        # don't save/load locker or things get ugly
        for k in (SKIP_SAVE_LOAD + SKIP_LOAD):
            if k in saved_dict:
                saved_dict.pop(k)
        self.__dict__.update(saved_dict)  # instead of overwriting class dict
        fh.close()

    def _save(self):
        self.nsave = self.get('nsave', 0) + 1

        # don't save/load locker/job_pkl or problems arise
        tosave = {k: v for k, v in self.__dict__.items()
                  if k not in SKIP_SAVE_LOAD}

        ufile.safe_save_pickle(tosave, self.pkl)

Experiment = BaseExperiment
