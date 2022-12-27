from typing import Union

from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm

from quapy.data import LabelledCollection
from quapy.method.aggregative import ACC, CC, _training_helper


class CCWithThreshold(CC):
    """
    The most basic Quantification method. One that simply classifies all instances and counts how many have been
    attributed to each of the classes in order to compute class prevalence estimates.

    :param learner: a sklearn's Estimator that generates a classifier
    """

    def __init__(self, learner: BaseEstimator, threshold: float):
        self.learner = learner
        self.threshold = threshold

    def classify(self, data):
        if hasattr(self.learner, "predict_proba"):
            y_scores = self.learner.predict_proba(data)[:,1]
            y_hat = np.where(y_scores < self.threshold, 0, 1)
        else:
            y_hat = self.learner.predict(data)
        return y_hat


class ACCWithThreshold(ACC):
    """
    `Adjusted Classify & Count <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_,
    the "adjusted" variant of :class:`CC`, that corrects the predictions of CC
    according to the `misclassification rates`.

    :param learner: a sklearn's Estimator that generates a classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set in which the
        misclassification rates are to be estimated.
        This parameter can be indicated as a real value (between 0 and 1, default 0.4), representing a proportion of
        validation data, or as an integer, indicating that the misclassification rates should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).

    Use a probabilistic classifier (like lightgbm) but with custom threshold decision.
    """

    def __init__(self, learner: BaseEstimator, threshold: float):
        self.learner = learner
        self.threshold = threshold

    def classify(self, data):
        if hasattr(self.learner, "predict_proba"):
            y_scores = self.learner.predict_proba(data)[:,1]
            y_hat = np.where(y_scores < self.threshold, 0, 1)
        else:
            y_hat = self.learner.predict(data)
        return y_hat
    
    def fit(self, data: LabelledCollection, fit_learner=True, val_split: Union[float, int, LabelledCollection] = None):
        """
        Trains a ACC quantifier.

        :param data: the training set
        :param fit_learner: set to False to bypass the training (the learner is assumed to be already fit)
        :param val_split: either a float in (0,1) indicating the proportion of training instances to use for
            validation (e.g., 0.3 for using 30% of the training set as validation data), or a LabelledCollection
            indicating the validation set itself, or an int indicating the number `k` of folds to be used in `k`-fold
            cross validation to estimate the parameters
        :return: self
        """
        if val_split is None:
            val_split = self.val_split
            classes = data.classes_
        if isinstance(val_split, int):
            assert fit_learner == True, \
                'the parameters for the adjustment cannot be estimated with kFCV with fit_learner=False'
            # kFCV estimation of parameters
            y, y_ = [], []
            kfcv = StratifiedKFold(n_splits=val_split)
            pbar = tqdm(kfcv.split(*data.Xy), total=val_split)
            for k, (training_idx, validation_idx) in enumerate(pbar):
                pbar.set_description(f'{self.__class__.__name__} fitting fold {k}')
                training = data.sampling_from_index(training_idx)
                validation = data.sampling_from_index(validation_idx)
                learner, val_data = _training_helper(self.learner, training, fit_learner, val_split=validation)
                y_.append(learner.predict(val_data.instances))
                y.append(val_data.labels)

            y = np.concatenate(y)
            y_ = np.concatenate(y_)
            class_count = data.counts()
            classes = data.classes_

            # fit the learner on all data
            self.learner, _ = _training_helper(self.learner, data, fit_learner, val_split=None)

        else:
            self.learner, val_data = _training_helper(self.learner, data, fit_learner, val_split=val_split)
            if hasattr(self.learner, "predict_proba"):
                y_scores = self.learner.predict_proba(val_data.instances)[:,1]
                y_ = np.where(y_scores < self.threshold, 0, 1)
            else:
                y_ = self.learner.predict(val_data.instances)
            
            y = val_data.labels
            classes = val_data.classes_

        self.cc = CCWithThreshold(self.learner, self.threshold)

        self.Pte_cond_estim_ = self.getPteCondEstim(classes, y, y_)

        return self
