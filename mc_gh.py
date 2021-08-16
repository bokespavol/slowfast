import numpy as np
from scipy.sparse import spdiags
from scipy.special import comb, factorial

class SlowFastModel:
    def __init__(self, xi, eta, f, fx, f1):
        """
        Define a slow-fast model
        
        xi and eta: vector of stoichiometries
        f, fx, f1: the rescaled reaction rates, its derivative, and correction.
        
        see the __init__ methods of the daughter classes for usage. 
        """
        self.xi = xi
        self.eta = eta
        self.f = f
        self.fx = fx
        self.f1 = f1
        self.Xname = "X"
        self.sname = "s"
        
    def A(self, x,th, der=(0,0), leading_order=True):
        """
        Return the Hamiltonian matrix
        
        Notes:
        If leading_order
            derivative[0] (w.r.t. x) must be 0 or 1
            derivative[1] (w.r.t. th) must be 0,1, or 2
        If not leading_order
            derivative must be (0,0)
        """
        smax = self.f(1.0).shape[1] - 1
        A = 0.
        for j in range(len(self.xi)):
            if leading_order:
                if not der[0]:
                    diag = self.f(x)[j,:]
                else:
                    diag = self.fx(x)[j,:]
            else:
                diag = self.f1(x)[j,:]                            
            A += spdiags([-diag*(der[1] == 0)],[0], smax+1,smax+1)
            A += spdiags([np.exp(self.xi[j]*th)*(self.xi[j]**der[1])*diag], 
                          [-self.eta[j]], smax+1, smax+1)
        return A.toarray()

class DelayedFeedback(SlowFastModel):
    def __init__(self, a0, a1, sthresh, b, smax):
        zeros = np.zeros(smax+1)
        s = np.arange(smax+1)
        a = a0*(s < sthresh) + a1*(s >= sthresh)
        """Order of reactions: (1) Decay; (2) Activation; (3) Activation'; (4) Production."""
        xi = np.array([  0, -1, -1, b])
        eta = np.array([-1,  1,  0, 0])  
        f = lambda x: np.array([s, x*(s < smax), x*(s == smax), a])
        fx = lambda x: np.array([zeros, (s < smax).astype('float'), (s == smax).astype('float'), zeros])
        f1 = lambda x: np.array([zeros, zeros, zeros, zeros])
        SlowFastModel.__init__(self, xi, eta, f, fx, f1)

class GeneAutoregulation(SlowFastModel):
    def __init__(self, a0, a1, n, omega, b, sqstr=True):
        """Order of reactions: (1) Binding; (2) Unbinding; (3) Decay; (4) Production."""
        xi = np.array([-n*sqstr, n*sqstr, -1, b])
        eta = np.array([1, -1, 0, 0])
        f = lambda x: np.array([[omega*x**n, 0], [0, omega], [x, x], [a0, a1]])
        fx = lambda x: np.array([[omega*n*x**(n-1), 0], [0,0], [1, 1], [0, 0]])
        f1 = lambda x: np.array([[-0.5*omega*(n-1)*n*x**(n-1), 0], [0,0], [0,0], [0,0]])        
        SlowFastModel.__init__(self, xi, eta, f, fx, f1)





























