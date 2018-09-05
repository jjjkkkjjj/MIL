import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from MIL.svm import LenearProblem
import inspect


class MILES(LenearProblem):
    def __init__(self, optimization=1, positive=1, negative=-1, lamb=1, mu=0.2, similarity='rbf',
                 kernel='pointless', C=1.0, p=3, gamma=1e0, scale_C=True, verbose=True, tol=1e-7, max_iter=-1):

        if kernel != 'pointless':
            print("Warning: MILES can't use kernel function. You shoule choose similarity")
        super(MILES, self).__init__(kernel, C, p, gamma, scale_C, verbose, tol, max_iter)

        self.similarity = similarity
        self.optimization = optimization
        self.positive = positive
        self.negative = negative
        self.lamb = lamb
        self.mu = mu

    def fit(self, bags, labels):
        # sort [p,p,p,p,n,n,n,n,n,n,n,n,n]
        Pbags_indices_of_bags = np.where(np.array(labels) == self.positive)[0]  # .shape[0] means l+
        Nbags_indices_of_bags = np.where(np.array(labels) == self.negative)[0]  # .shape means l-

        newbags, newlabels = [], []
        for indexPositive in Pbags_indices_of_bags:
            newbags.append(bags[indexPositive])
            newlabels.append(self.positive)
        for indexNegative in Nbags_indices_of_bags:
            newbags.append(bags[indexNegative])
            newlabels.append(self.negative)
        newlabels = np.array(newlabels)
        # mil to sil
        # X -> line up all instances in all bags (instance num, features dim) shape(n, feature dim)
        self.X_, y = self._sil(newbags, newlabels)

        # mapping
        number_of_bags = len(newbags)
        n = self.X_.shape[0] # number of instances


        l_plus = Pbags_indices_of_bags.shape[0]
        l_minus = Nbags_indices_of_bags.shape[0] # note that the number of bags is l_plus + l_minus

        self.s_ = self._similarity(newbags, self.X_) # s_.shape(number of Bags, k)

        """
        min lamb*sig(u+v) + mu*sig(xi) + (1-mu)*sig(eta) + 0*b
        s.t. leaving off
        
        """
        c = np.hstack(([self.lamb]*2*n,
                       [self.mu]*l_plus,
                       [1 - self.mu]*l_minus,
                       [0.0]))

        # sort [p,p,p,p,n,n,n,n,n,n,n,n,n]
        #self.s_ = self.s_[np.hstack((Pbags_indices_of_bags, Nbags_indices_of_bags))]

        """
         G =   u    v  xi  eta  b 
             (-m+  m+  -E   O  -1)
             ( m- -m-   O  -E   1)
             (       -E         0)
        """
        m_plus = self.s_[:l_plus]
        m_minus = self.s_[l_plus:]
        M = np.vstack((np.hstack((-m_plus, m_plus)),
                       np.hstack((m_minus, -m_minus))))

        G = np.vstack((np.hstack((M, -np.eye(number_of_bags), np.matrix(np.hstack(([-1.0]*l_plus, [1.0]*l_minus))).T)),
                       np.hstack((-np.eye(2*n + number_of_bags),
                                  np.matrix(np.array([0.0]*(2*n + number_of_bags))).T))))

        h = np.hstack(([-1.0]*number_of_bags,
                       [0.0]*(2*n + number_of_bags)))

        super(MILES, self).solve(c, G, h)

        #print(self.solve_['x'])
        parameters = np.squeeze(self.solve_['x'])
        u = parameters[:n]
        v = parameters[n:2*n]

        #u[u < self.tol] = 0
        #v[v < self.tol] = 0
        self.w_ = u - v

        self.nonzero_w_indices_ = np.where(np.abs(self.w_) > self.tol)[0] # I
        self.nonzero_w_ = self.w_[self.nonzero_w_indices_] # w*
        self.b_ = parameters[-1] # b*

        return self

    def predict(self, bags, instancePrediction=True):
        check_is_fitted(self, ['s_', 'w_', 'nonzero_w_', 'nonzero_w_indices_','b_', 'X_'])

        bag_predictions = []
        # bag prediction
        positive_indices = []

        s_ = self._similarity(bags, self.X_[self.nonzero_w_indices_]) # the k of w = 0 are pointless s_.shape(number of Bags, k)
        #print(s_.shape)

        for i, bag in enumerate(bags):
            #print(self.nonzero_w_.T.shape , s_[i].T.shape)
            f = float(self.nonzero_w_.T * s_[i].T) + self.b_
            bag_predictions.append(f)
        #print(bag_predictions)
        if not instancePrediction:
            return np.array(bag_predictions)
        else:
            pass


    def get_params(self, deep=True):
        super_args = super(MILES, self).get_params(deep=True)
        args, _, _, _ = inspect.getargspec(self.__init__)
        args.pop(0)
        super_args.update({key: getattr(self, key, None) for key in args})

        return super_args

    def _sil(self, bags, labels):
        sbags = [np.asmatrix(bag) for bag in bags]
        slabels = np.asmatrix(labels).reshape((-1, 1))

        X = np.vstack(sbags)
        y = np.vstack([float(cls) * np.matrix(np.ones((len(bag), 1)))
                                                    for bag, cls in zip(sbags, slabels)])

        return X, y

    def _similarity(self, bags, Y): # this method is like kernel function
        if self.similarity == 'rbf':
            """
            m = (min_j|x1j - X1|, min_j|x1j - X2|, ... , min_j|x1j - Xn|) 
                (min_j|x2j - X1|, min_j|x2j - X2|, ... , min_j|x2j - Xn|)
                ...
                (min_j|x(l+ + l-)j - X1|, min_j|x(l+ + l-)j - X2|, ... , min_j|x(l+ + l-)j - Xn|)
             
            note that j for each bag is different size
            """
            m = []
            for bag in bags:
                d = [] # feature vector of a bag
                for k in range(Y.shape[0]):
                    #print(bag.shape)
                    #print(((bag - Y[k]) * (bag - Y[k]).T).shape)
                    """
                    (bag - Y[k]) * (bag - Y[k]).T is like a correlation matrix 
                    i want a norm only so i take it from matrix by diag
                    
                    """
                    d.append(np.min(np.diag((bag - Y[k]) * (bag - Y[k]).T)))
                m.append(d)
            m = np.asmatrix(np.array(m))
            return np.exp(-self.gamma * m)
        else:
            raise ValueError("{0} is invalid similarity".format(self.similarity))