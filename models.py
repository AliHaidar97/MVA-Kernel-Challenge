from scipy import optimize
from scipy.linalg import cho_factor, cho_solve
import numpy as np
import cvxopt
import cvxopt.solvers

def to_binary(y):
    return ((y + 1) / 2).astype(int)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class KernelSVC:
    
    def __init__(self, C, epsilon = 1e-3):
        self.type = 'non-linear'
        self.C = C                                     
        self.alpha = None
        self.epsilon = epsilon
        self.norm_f = None
        self.a = None
       
    
    def fit(self, K_train, y):
        
        y = np.array(y)
       
        #### You might define here any variable needed for the rest of the code
        N = len(y)
        
        K_train += 1   
        
        # Set up quadratic programming problem
        P = cvxopt.matrix(np.outer(y, y) * K_train)
        q = cvxopt.matrix(-1 * np.ones(N))
        G = cvxopt.matrix(np.vstack((-1 * np.eye(N), np.eye(N))))
        h = cvxopt.matrix(np.hstack((np.zeros(N), self.C * np.ones(N))))
        A = cvxopt.matrix(y.reshape(1, -1)) * 1.0
        b = cvxopt.matrix(np.zeros(1))
        # Solve the quadratic program using cvxopt       
        cvxopt.solvers.options['show_progress'] = True
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        self.alpha = np.ravel(solution['x'])
        
        
        #clip
        self.alpha[self.alpha < 1e-5] = 0
        ## Assign the required attributes
        self.a = np.diag(y)@self.alpha 
        f = K_train@self.a
        mask = ((self.alpha < self.C) & (self.alpha > 0))
        self.b =  np.median((1 - y[mask]*f[mask])/y[mask]) #''' -----------------offset of the classifier------------------ '''
        self.norm_f = self.a.T@K_train@self.a   #'''------------------------RKHS norm of the function f ------------------------------'''
       

    ### Implementation of the separting function $f$ 
    def separating_function(self, K_test):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        K_test += 1
        return K_test@self.a
    
    
    def predict(self, K_test):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(K_test)
        return 2 * ((d+self.b)> 0) - 1
    
    def predict_proba(self, X):
        d = self.separating_function(X)
        return sigmoid(d + self.b)
    
    
