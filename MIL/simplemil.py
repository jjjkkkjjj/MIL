import numpy as np
from sklearn.svm.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn import svm

class SimpleMIL(BaseEstimator, ClassifierMixin):
    def __init__(self, method='average'):
        self.method = method

    def fit(self, bags, labels):

        if self.method == 'average':
            bag_mean = np.asarray([np.mean(bag, axis=0) for bag in bags])
            bag_modified = bag_mean
        elif self.method == 'extreme':
            bag_max = np.asarray([np.amax(bag, axis=0) for bag in bags])
            bag_min = np.asarray([np.amin(bag, axis=0) for bag in bags])
            bag_extreme = np.concatenate((bag_max, bag_min), axis=1)
            bag_modified = bag_extreme
        elif self.method == 'max':
            bag_max = np.asarray([np.amax(bag, axis=0) for bag in bags])
            bag_modified = bag_max
        elif self.method == 'min':
            bag_min = np.asarray([np.amin(bag, axis=0) for bag in bags])
            bag_modified = bag_min
        else:
            raise ValueError("{0} is invalid method name".format(self.method))

        self.model_ = svm.SVC()
        self.model_.fit(bag_modified, labels)

        return self

    def predict(self, bags):
        check_is_fitted(self, 'model_')

        if self.method == 'average':
            bag_mean = np.asarray([np.mean(bag, axis=0) for bag in bags])
            bag_modified = bag_mean
        elif self.method == 'extreme':
            bag_max = np.asarray([np.amax(bag, axis=0) for bag in bags])
            bag_min = np.asarray([np.amin(bag, axis=0) for bag in bags])
            bag_extreme = np.concatenate((bag_max, bag_min), axis=1)
            bag_modified = bag_extreme
        elif self.method == 'max':
            bag_max = np.asarray([np.amax(bag, axis=0) for bag in bags])
            bag_modified = bag_max
        elif self.method == 'min':
            bag_min = np.asarray([np.amin(bag, axis=0) for bag in bags])
            bag_modified = bag_min
        else:
            raise ValueError("{0} is invalid method name".format(self.method))

        predictions = self.model_.predict(bag_modified)
        return predictions
