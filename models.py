from scipy import optimize
from scipy.linalg import cho_factor, cho_solve
import numpy as np

class KernelSVC:
    
    def __init__(self, C, epsilon = 1e-3):
        self.type = 'non-linear'
        self.C = C                                     
        self.alpha = None
        self.epsilon = epsilon
        self.norm_f = None
        self.a = None
       
    
    def fit(self, K_train, y):
       #### You might define here any variable needed for the rest of the code
        N = len(y)
        
        # Lagrange dual problem
        def loss(alpha):
            #'''--------------dual loss ------------------ '''
            return (1/2)*((np.diag(y)@alpha.reshape(-1,1)).T@K_train@(np.diag(y)@alpha.reshape(-1,1))) - np.sum(alpha)
        
        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            # '''----------------partial derivative of the dual loss wrt alpha -----------------'''
            return np.diag(y)@K_train@np.diag(y)@alpha.reshape(-1,1) - 1
        
        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0
        alpha = np.ones(N)
        fun_eq = lambda alpha:  alpha@y # '''----------------function defining the equality constraint------------------'''        
        jac_eq = lambda alpha:   y #'''----------------jacobian wrt alpha of the  equality constraint------------------'''
        fun_ineq = lambda alpha:   np.hstack(((self.C - alpha),alpha)) # '''---------------function defining the inequality constraint-------------------'''     
        jac_ineq = lambda alpha:   np.vstack((-np.eye(alpha.shape[0]),np.eye(alpha.shape[0])))# '''---------------jacobian wrt alpha of the  inequality constraint------------------- '''
        constraints = ({'type': 'eq',  'fun': fun_eq, 'jac':jac_eq},
                       {'type': 'ineq', 
                        'fun': fun_ineq,
                         'jac': jac_ineq  
                        })

        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.zeros(N), 
                                   method='SLSQP', 
                                   jac=lambda alpha: grad_loss(alpha), 
                                   constraints=constraints)
        self.alpha = optRes.x
        #clip
        self.alpha[self.alpha < 1e-5] = 0
        ## Assign the required attributes
        self.a = np.diag(y)@self.alpha 
        f = K_train@self.a
        mask = ((self.alpha < self.C) & (self.alpha > 0))
        mask = (self.alpha < self.C)
        self.b =  np.mean((1 - y[mask]*f[mask])/y[mask]) #''' -----------------offset of the classifier------------------ '''
        self.norm_f = self.a.T@K_train@self.a   #'''------------------------RKHS norm of the function f ------------------------------'''
       

    ### Implementation of the separting function $f$ 
    def separating_function(self, K_test):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        return K_test@self.a
    
    
    def predict(self, K_test):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(K_test)
        return 2 * (d+self.b> 0) - 1
    
