import sys, os
#os.chdir(os.path.abspath('__file__'))
sys.path.append(os.path.realpath('..'))
from MIL.simplemil import SimpleMIL
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

simpleMIL = SimpleMIL(method='average')
scores = model_selection.cross_val_score(simpleMIL, bags, labels, cv=5) # label's shape is required for (a,) instead of (a, 1)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
"""
train_bags, test_bags, train_labels, test_labels = cross_validation.train_test_split(bags, labels, test_size=0.1, random_state=seed)

simpleMIL = SimpleMIL(method='average')
simpleMIL.fit(train_bags, train_labels)
predictions = simpleMIL.predict(test_bags)
accuracy = np.average(test_labels.T == np.sign(predictions))
print ('\n Accuracy: %.2f%%' % (100 * accuracy))
fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions, pos_label=1.)
metrics.auc(fpr, tpr)
"""