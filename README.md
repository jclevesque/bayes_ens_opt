
# Bayesian Hyperparameter Optimization for Ensemble Learning

This is an application of Bayesian optimization to ensemble optimization. This code is from the paper titled "Bayesian Hyperparameter Optimization for Ensemble Learning" presented at the 32nd UAI conference in 2016 [[pdf](https://arxiv.org/pdf/1605.06394)]. The goal is to train an ensemble as we optimize the hyperparameters of a given model, or multiple models simultaneously. The overhead with regards to a single classifier hyperparameter optimization is the computation of a majority vote with all the classifiers that were trained so far. That amounts to O(tmn), where t is the number of classifiers that were trained so far (will grow as iterations pass), m is the number of classifiers in the desired ensemble (chosen beforehand) and n is the number of cross-validation data points. 

I have tinkered quite a bit with the code to remove useless research artifacts, so it could not work as advertised. Double check results and report anomalies if you find them.

Comes with no guarantees and will run quite slow for large search spaces (because of the underlying Gaussian Process model, which is not very good in high dimensional spaces anyway). This code is also absolutely not meant for very large datasets since everything is kept in memory. A solution for larger datasets would be to use pointers to samples instead of copies, but this toolbox does not support it at the moment (and probably never will). 

