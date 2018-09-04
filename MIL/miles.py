import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from MIL.svm import LenearProblem



class MILES(LenearProblem):
    def __init__(self, optimization=1, sigma=1, similarity='gaussian', negative=-1, lamb=1, mu=0.2,
                 kernel='linear', C=1.0, p=3, gamma=1e0, scale_C=True, verbose=True, tol=1e-7, max_iter=-1):

        super(MILES, self).__init__(kernel, C, p, gamma, scale_C, verbose, tol, max_iter)

        self.optimization = optimization
        self.sigma = sigma
        self.similarity = similarity
        self.negative = negative
        self.lamb = lamb
        self.mu = mu

    def fit(self, bags, labels):

        # mil to sil
        # X -> line up all instances in all bags (instance num, features dim)
        self.X_, y = self._sil(bags, labels)

        # mapping
        number_of_bags = len(bags)
        n = self.X_.shape[0] # number of instances

        Pbags_indices_of_bags = np.where(np.array(labels) == 1)[0] # .shape[0] means l+
        Nbags_indices_of_bags = np.where(np.array(labels) == self.negative)[0]  # .shape means l-

        l_plus = Pbags_indices_of_bags.shape[0]
        l_minus = Nbags_indices_of_bags.shape[0] # note that the number of bags is l_plus + l_minus

        d = []
        for bag in bags:
            #print(type(bag).__name__)
            d_ = []
            for k in range(self.X_.shape[0]):
                d_.append(np.min(np.linalg.norm(bag - self.X_[k, :], axis=1)))
            d.append(d_)

        d = np.asmatrix(np.array(d))
        #print(d)
        self.s_ = self._similarity(d)

        """
        min lamb*sig(u+v) + mu*sig(xi) + (1-mu)*sig(eta) + 0*b
        s.t. leaving off
        
        """
        c = np.hstack(([self.lamb]*2*n,
                       [self.mu]*l_plus,
                       [1 - self.mu]*l_minus,
                       [0]))

        # sort [p,p,p,p,n,n,n,n,n,n,n,n,n]
        self.s_ = self.s_[np.hstack((Pbags_indices_of_bags, Nbags_indices_of_bags))]
        """
         G =   u    v  xi  eta  b 
             (-m+  m+  -E   O  -1)
             (-m-  m-   O  -E  -1)
             (       -E         0)
        """
        G = np.vstack((np.hstack((-self.s_, self.s_, -np.eye(number_of_bags), np.matrix(np.array([-1]*number_of_bags)).T)),
                       np.hstack((-np.eye(2*n + number_of_bags),
                                  np.matrix(np.array([0]*(2*n + number_of_bags))).T))))

        h = np.hstack(([-1.0]*number_of_bags,
                       [0]*(2*n + number_of_bags)))

        super(MILES, self).solve(c, G, h)

        u = self.solve_['x'][:n]
        v = self.solve_['x'][n:2*n]
        w_ = u - v
        self.nonozerow_indices_ = np.where(w_ > 0)[0]
        self.b = self.solve_['x'][-1]

        #print(self.solve_['x'])
        return


    def predict(self, bags, instancePrediction=True):
        check_is_fitted(self, ['s_', 'w_', 'b_'])


        for bag in bags:
            U = []
            for k in self.nonozerow_indices_:
                U.append(np.argmin(np.linalg.norm(bag - self.X_[k, :])))
                
        pass

    def _sil(self, bags, labels=None):
        sbags = [np.asmatrix(bag) for bag in bags]
        slabels = np.asmatrix(labels).reshape((-1, 1))

        X = np.vstack(sbags)
        y = np.vstack([float(cls) * np.matrix(np.ones((len(bag), 1)))
                       for bag, cls in zip(sbags, slabels)])

        return X, y

    def _similarity(self, d):
        if self.similarity == 'gaussian':
            return np.exp(-np.power(d, 2) / self.sigma*self.sigma)
        else:
            raise ValueError("{0} is invalid similarity".format(self.similarity))