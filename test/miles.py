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
bags, labels, X = load_data('musk1_original')
train_bags, test_bags, train_labels, test_labels = model_selection.train_test_split(bags, labels, test_size=0.1, random_state=seed)

labels = labels.reshape(labels.shape[0],)

miles = MILES(negative=0, gamma=1.0/500000, kernel='rbf', lamb=0.45, mu=0.5)
miles.fit(train_bags, train_labels)
predictions = miles.predict(test_bags, instancePrediction=False)
test_labels = np.array(test_labels, dtype=int)
test_labels[test_labels == 0] = -1
print(test_labels)
print(np.sign(predictions))
accuracy = np.average(test_labels.T == np.sign(predictions))
print ('\n Accuracy: %.2f%%' % (100 * accuracy))