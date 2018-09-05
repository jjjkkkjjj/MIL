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
from sklearn.metrics import accuracy_score
from MIL.misvmio import parse_c45, bag_set

#check_estimator(SimpleMIL)
"""
seed = 66
bags, labels, X = load_data('musk1_original')
train_bags, test_bags, train_labels, test_labels = model_selection.train_test_split(bags, labels, test_size=0.1, random_state=seed)

labels = labels.reshape(labels.shape[0],)
"""
# Load list of C4.5 Examples
example_set = parse_c45('musk1')

# Group examples into bags
bagset = bag_set(example_set)

# Convert bags to NumPy arrays
# (The ...[:, 2:-1] removes first two columns and last column,
#  which are the bag/instance ids and class label)
bags = [np.array(b.to_float())[:, 2:-1] for b in bagset]
labels = np.array([b.label for b in bagset], dtype=float)

train_bags = bags[10:]
train_labels = labels[10:]
test_bags = bags[:10]
test_labels = labels[:10]


miles = MILES(negative=0, gamma=1.0/500000, similarity='rbf', lamb=0.45, mu=0.5)
miles.fit(train_bags, train_labels)
"""
predictions = miles.predict(train_bags, instancePrediction=False)
# fot train
train_labels = np.array(train_labels, dtype=int)
train_labels[train_labels == 0] = -1
print(train_labels)
print(np.sign(predictions))
accuracy = np.average(train_labels.T == np.sign(predictions))
print ('\n Accuracy: %.2f%%' % (100 * accuracy))
exit()


# for test
predictions = miles.predict(test_bags, instancePrediction=False)
test_labels = np.array(test_labels, dtype=int)
test_labels[test_labels == 0] = -1
print(test_labels)
print(np.sign(predictions))
accuracy = np.average(test_labels.T == np.sign(predictions))
print ('\n Accuracy: %.2f%%' % (100 * accuracy))
"""
# cross validation
def my_scorer(estimator, x, y):
    y = np.array(y, dtype=int)
    y[y == 0] = -1
    yPred = np.sign(estimator.predict(x, False))
    a = np.average(y.T == yPred)
    return a

miles = MILES(negative=0, gamma=1.0/500000, similarity='rbf', lamb=0.45, mu=0.5)
scores = model_selection.cross_val_score(miles, bags, labels, scoring=my_scorer, cv=5) # label's shape is required for (a,) instead of (a, 1)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
