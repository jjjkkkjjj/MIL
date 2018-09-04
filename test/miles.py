import sys, os
#os.chdir(os.path.abspath('__file__'))
sys.path.append(os.path.realpath('..'))
from MIL.miles import MILES
from MIL.load_data import load_data
from sklearn.utils import shuffle
from sklearn.utils.estimator_checks import check_estimator
import random as rand
from sklearn import model_selection, metrics
import numpy as np

#check_estimator(SimpleMIL)

seed = 66
bags, labels, X = load_data('musk1_scaled')

labels = labels.reshape(labels.shape[0],)

miles = MILES(negative=0)
miles.fit(bags, labels)