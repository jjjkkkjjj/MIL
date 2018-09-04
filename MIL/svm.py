import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import cvxopt
from scipy.spatial.distance import cdist
import inspect

class SVM(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel='linear', C=1.0, p=3, gamma=1e0, scale_C=True,
                 verbose=True, tol=1e-7, max_iter=-1):
        self.kernel = kernel
        self.C = C
        self.p = p
        self.gamma = gamma
        self.scale_C = scale_C
        self.verbose = verbose
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, **kwargs):
        pass


    def predict(self, **kwargs):
        #check_is_fitted(self, ['coef_'])
        pass

    def _kernel(self, X, Y):
        if self.kernel == 'rbf':
            #print(X.shape, Y.shape)
            K = -self.gamma * cdist(X, Y, 'sqeuclidean')

            return np.exp(K)
        else:
            raise ValueError("{0} is invalid kernel".format(self.kernel))


class LenearProblem(SVM):
    def __init__(self, kernel='linear', C=1.0, p=3, gamma=1e0, scale_C=True,
                 verbose=True, tol=1e-7, max_iter=-1):
        super(LenearProblem, self).__init__(kernel, C, p, gamma, scale_C,
                 verbose, tol, max_iter)

    """ 
    solving lenear problem using cvxopt
    @:param c : coefficient of variables which we want to solve　row vector...i don't know why
    @:param G : the coefficient matrix of inequality constraint
    @:param h : constraint row vector of inequality constraint
    @:param A : the coefficient matrix of equality constraint
    @:param b : constraint row vector of equality constraint
    """
    def solve(self, c, G=None, h=None, A=None, b=None):

        if G is not None and h is not None and A is not None and b is not None:
            self.solve_ = cvxopt.solvers.lp(cvxopt.matrix(c), cvxopt.matrix(G), cvxopt.matrix(h), cvxopt.matrix(A), cvxopt.matrix(b))

        elif G is not None and h is not None:
            self.solve_ = cvxopt.solvers.lp(cvxopt.matrix(c), cvxopt.matrix(G), cvxopt.matrix(h))

        elif A is not None and b is not None:
            self.solve_ = cvxopt.solvers.lp(cvxopt.matrix(c), cvxopt.matrix(A), cvxopt.matrix(b))

        else:
            raise ValueError("right combination are (G, h, A, b), (G, h), (A, b)")

        return self.solve_

    def get_params(self, deep=True):
        args, _, _, _ = inspect.signature(super(LenearProblem, self).__init__)
        args.pop(0)
        return {key: getattr(self, key, None) for key in args}