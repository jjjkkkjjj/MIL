import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from MIL.svm import LenearProblem
import inspect


class MILES(LenearProblem):
    def __init__(self, optimization=1, negative=-1, lamb=1, mu=0.2,
                 kernel='linear', C=1.0, p=3, gamma=1e0, scale_C=True, verbose=True, tol=1e-7, max_iter=-1):

        super(MILES, self).__init__(kernel, C, p, gamma, scale_C, verbose, tol, max_iter)

        self.optimization = optimization
        self.negative = negative
        self.lamb = lamb
        self.mu = mu

    def fit(self, bags, labels):

        # mil to sil
        # X -> line up all instances in all bags (instance num, features dim) shape(n, feature dim)
        self.X_, y = self._sil(bags, labels)

        # mapping
        number_of_bags = len(bags)
        n = self.X_.shape[0] # number of instances

        Pbags_indices_of_bags = np.where(np.array(labels) == 1)[0] # .shape[0] means l+
        Nbags_indices_of_bags = np.where(np.array(labels) == self.negative)[0]  # .shape means l-

        l_plus = Pbags_indices_of_bags.shape[0]
        l_minus = Nbags_indices_of_bags.shape[0] # note that the number of bags is l_plus + l_minus
        """
        d = []
        for bag in bags:
            #print(type(bag).__name__)
            d_ = []
            for k in range(n):
                d_.append(np.min(np.linalg.norm(bag - self.X_[k, :], axis=1)))
            d.append(d_)

        d = np.asmatrix(np.array(d))
        #print(d)
        self.s_ = np.exp(self.gamma*np.power(d, 2))
        print(self.s_.shape)
        """
    
        s = []
        for bag in bags:
            s_ = []
            for k in range(n):
                j = np.argmax([np.inner(bag[j_, np.newaxis], self.X_[k, :]) for j_ in range(len(bag))])
                s_.append(self._kernel(bag[j, np.newaxis], self.X_[k, :]))
            s.append(s_)
        self.s_ = np.asmatrix(np.array(s))

        """
        min lamb*sig(u+v) + mu*sig(xi) + (1-mu)*sig(eta) + 0*b
        s.t. leaving off
        
        """
        c = np.hstack(([self.lamb]*2*n,
                       [self.mu]*l_plus,
                       [1 - self.mu]*l_minus,
                       [0.0]))

        # sort [p,p,p,p,n,n,n,n,n,n,n,n,n]
        self.s_ = self.s_[np.hstack((Pbags_indices_of_bags, Nbags_indices_of_bags))]
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
        parameters = np.array(self.solve_['x'])
        u = parameters[:n]
        v = parameters[n:2*n]

        self.w_ = u - v
        self.nonzero_w_indices_ = np.where(self.w_ > 0)[0] # I
        self.nonzero_w_ = self.w_[self.nonzero_w_indices_] # w*
        self.b_ = self.solve_['x'][-1] # b*



    def predict(self, bags, instancePrediction=True):
        check_is_fitted(self, ['s_', 'w_', 'nonzero_w_', 'nonzero_w_indices_','b_'])

        bag_predictions = []
        # bag prediction
        positive_indices = []
        for i, bag in enumerate(bags):
            f = []
            for j in range(bag.shape[0]):
                #print(self._kernel(bag[j, np.newaxis], self.X_[self.nonzero_w_indices_, :]))
                f.append(np.dot(self._kernel(bag[j, np.newaxis], self.X_[self.nonzero_w_indices_, :]),
                                self.nonzero_w_))

            bag_predictions.append(np.sum(f) + self.b_)
            if bag_predictions[i] > 0:
                positive_indices.append(i)
        print(bag_predictions)
        if not instancePrediction:
            return bag_predictions
        else:

            pass



        pass

    def get_params(self, deep=True):
        super_args = super(MILES, self).get_params(deep=True)
        args, _, _, _ = inspect.signature(self.__init__)
        args.pop(0)
        super_args.update({key: getattr(self, key, None) for key in args})

        return super_args

    def _sil(self, bags, labels=None):
        sbags = [np.asmatrix(bag) for bag in bags]
        slabels = np.asmatrix(labels).reshape((-1, 1))

        X = np.vstack(sbags)
        y = np.vstack([float(cls) * np.matrix(np.ones((len(bag), 1)))
                       for bag, cls in zip(sbags, slabels)])

        return X, y
