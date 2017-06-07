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

import shutil

import numpy as np
import sklearn.datasets as skd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from beo import bo
from beo.spaces import beo_space
from beo.util import file_handling
from beo.util import trace

np.random.seed(42)
output_dir = 'output_ex_bo'

# clear previous results if any
try:
    shutil.rmtree(output_dir)
except:
    pass

# Load dataset and generate splits
dat = skd.load_digits()

# This is all rather inefficient, lots of data duplication esp. for k-fold, 
# but for medium sized dataset should not be a problem
D = {}
# tv_data is used for retraining on whole training+validation data,
# only required if eval_params['retrain'] = True
D['tv_data'], D['test_data'], D['tv_labels'], D['test_labels'] = \
    train_test_split(dat['data'], dat['target'])

train_data, val_data, train_labels, val_labels = train_test_split(
    D['tv_data'], D['tv_labels'])
D['train_data'] = [train_data]
D['train_labels'] = [train_labels]
D['val_data'] = [val_data]
D['val_labels'] = [val_labels]

# define a search space
search_params = [
    {'name':'C', 'type':'float', 'scale':'log', 'min':-5, 'max':5, 'size':1},
    {'name':'gamma', 'type':'float', 'scale':'log', 'min':-5, 'max':5, 'size':1}
] 
search_space = beo_space.SearchSpace(search_params)

# Parameters for the evaluation of the models, i.e. training model and
# computing its validation error
# Basically, those are the parameters sent to the train_test_learner function
eval_params = {'learner_class': SVC, 
    'learner_fixed_params': {},
    # Train / val / testing data
    'datasets': D,
    # Retrain on whole train+val data after job is completed
    'retrain': True,
    'distinct_process': True, 
    'crash_on_exception': True,
    # 60 seconds to evaluate cross-validation accuracy, this includes 
    # training as many models as there are validation folds + 1
    'max_eval_time': 60
    }

# Create the optimization object
optimizer = bo.BayesianOptimizer(search_space, 
    output_folder=output_dir,
    chooser_mod='beo.choosers.gp_ei_opt', 
    expt_mgr_mod='beo.expt_mgr.base',
    eval_params=eval_params,
    func_scale=50, # scale objective so that the noise of the GP is better fit
    chooser_params={},
    expt_mgr_params={},
    max_iters=30, 
    n_candidates=10000
    )

optimizer.optimize()

# Print some stats and information once the experiment is over
expt_trace = file_handling.load_pickle(output_dir + "/expt_trace.pkl.bz2")
trace_dict = trace.inside_out(expt_trace)
# single best according to the validation dataset
sing_best_i = np.argmin(trace_dict['val_T_err'])

print("Expt done, priting performance of best model at last iteration.")
print("Validation error: %f" 
    % (expt_trace[sing_best_i]['val_T_err']))

print("Testing error: %f" 
    % (expt_trace[sing_best_i]['test_TV_err']))
