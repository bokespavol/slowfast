import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import newton
from scipy.integrate import odeint, quad
from numpy.linalg import lstsq, svd, eig
from scipy.sparse import spdiags, linalg
from scipy.interpolate import interp1d

def myintegrate(x, f):
    """Find primitive function F such that F(x[0])=0"""
    N = len(f)
    F = np.ndarray(N)
    F[0] = 0.
    for i in range(1,N):
        F[i] = F[i-1] + 0.5*(f[i-1] + f[i])*(x[i] - x[i-1])
        
    return F

def myintegrate_from_midpoint(x, dF, i):
    """Find primitive function F such that F(x[i])=0"""
    dF1 = dF[i:]
    dF2 = dF[i::-1]
    x1 = x[i:]
    x2 = x[i::-1]
    F1 = myintegrate(x1, dF1)
    F2 = myintegrate(x2, dF2)
    return np.concatenate((F2[::-1], F1[1:]))

def make_positive(v):
    if np.sum(v) > 0:
        return v
    else:
        return -v

def get_princ(A):
    lams, V = eig(A)
    i = np.argmax(np.real(lams))
    lam = lams[i]
    if np.any(V[:,i].imag != 0.) or lam.imag != 0:
        print("Warning (Perron--Frobenius failure): nonzero imaginary parts")
    if np.any(V[:,i].real >= 0.) and np.any(V[:,i].real <= 0.):
        print("Warning (Perron--Frobenius failure): mixed sign eigenvector components")
    v = V[:,i].real
    return lam, v 
        
def get_princ_lr(A):
    lam, v = get_princ(A)
    lam_copy, u = get_princ(A.transpose())
    #if not lam == lam_copy:
    #    print("Warning: eigenvalues mismatch {} vs {}".format(lam,lam_copy))
    u = make_positive(u)
    v = make_positive(v)
    v = v/np.sum(v)
    u = u/np.dot(u,v)
    return u, lam, v

class PhasePlane:
    def __init__(self, A):
        self.A = A                
    
    def gradH(self, x,th):
        u, H, v = get_princ_lr(self.A(x,th))
        Hx = np.inner(u, self.A(x,th, der=(1,0)).dot(v))
        Hth = np.inner(u, self.A(x,th, der=(0,1)).dot(v))
        return Hx, Hth    
    
    def hamrhs(self, t, state):
        x = state[0]
        th = state[1]
        Hx, Hth = self.gradH(x, th)
        return [Hth, -Hx]
        
    def oderhs(self, th, x):
        Hx, Hth = self.gradH(x, th)
        return -Hx/Hth

    def Hth(self, x):
        """th=0"""
        if type(x) == np.ndarray and x.ndim == 1:
            res = np.ndarray(len(x))
            for i in range(len(x)):
                res[i] = self.gradH(x[i], 0)[1]
            return res
        else:
            return self.gradH(x, 0)[1]            
                  
    def rho(self, x):
        return get_princ_lr(self.A(x,0))[2]

    def Hthx(self, x):
        """th=0"""        
        rho = get_princ_lr(self.A(x,0))[2]
        rhoxtilde = lstsq(self.A(x,0), -self.A(x,0,der=(1,0)).dot(rho), rcond=None)[0]
        rhox = rhoxtilde - np.inner(np.ones(rhoxtilde.shape), rhoxtilde)*rho
        return np.inner(np.ones(rho.shape), self.A(x,0, der=(1,1)).dot(rho) 
                        + self.A(x,0, der=(0,1)).dot(rhox))
        
    def Hthth(self, x):
        """th=0"""
        v = get_princ_lr(self.A(x,0))[2]
        vthtilde = lstsq(self.A(x,0), (np.outer(v, np.ones(v.shape))  
                    -np.eye(v.shape[0])).dot(self.A(x,0, der=(0,1)).dot(v)), rcond=None)[0]
        vtilde = vthtilde - np.inner(np.ones(vthtilde.shape), vthtilde)*v
        return np.inner(np.ones(v.shape), self.A(x,0,der=(0,2)).dot(v) + 
                        2.*self.A(x,0,der=(0,1)).dot(vtilde))
        
    def get_dpsi(self, x, th, d2phi):
        u, H, v = get_princ_lr(self.A(x,th))
        sol = lstsq(self.A(x,th), -(d2phi*self.A(x,th, der=(0,1)) 
                        + self.A(x,th, der=(1,0))).dot(v), rcond=None)[0]
        dw = sol - np.dot(np.ones(sol.shape), sol)*v
        num = np.inner(u, self.A(x,th, der=(0,1)).dot(dw)
                + (0.5*d2phi*self.A(x,th, der=(0,2)) 
                    + self.A(x,th, der=(1,1)) 
                    - self.A(x, th, leading_order=False)).dot(v))
        denom = np.inner(u, self.A(x,th, der=(0,1)).dot(v))
        return num/denom

    def lnarhs(self, v, t):
        x = v[0]
        sigma2 = v[1]
        dxdt = self.Hth(x)
        dsigma2dt = 2.*self.Hthx(x)*sigma2 + self.Hthth(x)
        return [dxdt, dsigma2dt]    
    
    def solvelna(self, t, xinit):
        sol = odeint(self.lnarhs, [xinit, 0] , t)
        return sol[:, 0], sol[:, 1]
        
    
"""Routine for the calculation of the nontrivial heteroclinic connections"""

class WentzelKramsersBrillouin:
    def __init__(self, pp):
        self.pp = pp
        """Fixed pts"""
        self.fps = None
        self.lin_at_fps = None
        self.d2phi_at_fps = None
        """The Hamiltonian zero set"""
        self.x_vec = None
        self.phi_vec = None
        self.dphi_vec = None       
        self.d2phi_vec = None
        self.phi = None
        self.dphi = None
        self.d2phi = None
        """Prefactor"""
        self.x_sel = None
        self.dpsi_sel = None
        self.psi_sel = None
        self.psi = None
        self.d2psi = None

    def findfps(self, guesses):
        self.fps = []
        self.lin_at_fps = []
        self.d2phi_at_fps = []
        for guess in np.sort(guesses):
            fp = newton(lambda x: self.pp.Hth(x), guess, fprime = lambda x: self.pp.Hthx(x))
            self.fps.append(fp)
            self.lin_at_fps.append(self.pp.Hthx(fp))
            self.d2phi_at_fps.append(-2.*self.pp.Hthx(fp)/self.pp.Hthth(fp))

    def findpotential(self, xmin, xmax, pert=0.01, M=100):
        if self.fps is None:
            print("Run findfps first to find fixed points")
            return None
        xbash = [xmin]
        for i in range(len(self.fps)-1):
            xbash.append((self.fps[i]+self.fps[i+1])/2.)    
        xbash.append(xmax)    
        
        xpatch = []
        thpatch = []
        d2phipatch = []
        for i in range(len(self.fps)):
            xleft = np.linspace(self.fps[i] - pert, xbash[i], M)
            xright = np.linspace(self.fps[i] + pert, xbash[i+1], M)
            thleft = odeint(self.pp.oderhs, -pert*self.d2phi_at_fps[i], xleft).flatten()
            thright = odeint(self.pp.oderhs, pert*self.d2phi_at_fps[i], xright).flatten()
            d2phileft = np.ndarray(xleft.shape)
            d2phiright = np.ndarray(xright.shape)
            for j in range(len(xleft)):
                d2phileft[j] = self.pp.oderhs(thleft[j], xleft[j])
            for j in range(len(xright)):
                d2phiright[j] = self.pp.oderhs(thright[j], xright[j])
            xpatch = np.concatenate((xpatch, xleft[::-1], [self.fps[i]], xright[:-1]))
            thpatch = np.concatenate((thpatch, thleft[::-1], [0], thright[:-1]))
            d2phipatch = np.concatenate((d2phipatch, d2phileft[::-1], [self.d2phi_at_fps[i]], d2phiright[:-1]))
            if i == len(self.fps) - 1:
                xpatch = np.concatenate((xpatch, [xright[-1]]))
                thpatch = np.concatenate((thpatch, [thright[-1]]))
                d2phipatch = np.concatenate((d2phipatch, [d2phiright[-1]]))
        
        phipatch = myintegrate_from_midpoint(xpatch, thpatch, np.argmin(np.abs(xpatch - self.fps[int(len(self.fps)>1)])))
        self.x_vec = xpatch
        self.phi_vec = phipatch
        self.dphi_vec = thpatch
        self.d2phi_vec = d2phipatch
        self.phi = interp1d(xpatch, phipatch, kind='cubic')
        self.dphi = interp1d(xpatch, thpatch, kind='cubic')
        self.d2phi = interp1d(xpatch, d2phipatch, kind='cubic')
        
    def findprefactor(self, avoidfps=-1):
        if self.x_vec is None:
            print("Run findpotential first to find the WKB potential")
            return None   
        dpsi =np.ndarray(shape=self.x_vec.shape)
        for i in range(len(self.x_vec)):
            dpsi[i] = self.pp.get_dpsi(self.x_vec[i], self.dphi_vec[i], self.d2phi_vec[i])
                 
        sel = np.ones(len(self.x_vec), dtype=bool)
        for fp in self.fps:
            sel *= np.abs(self.x_vec - fp) > avoidfps
            
        x_sel = self.x_vec[sel]
                                    
        psi_sel = myintegrate_from_midpoint(x_sel, dpsi[sel], 
                np.argmin(np.abs(x_sel - self.fps[int(len(self.fps)>1)])))
        self.x_sel = x_sel
        self.dpsi_sel = dpsi[sel]
        self.psi_sel = psi_sel
        self.dpsi = interp1d(x_sel, dpsi[sel], kind='cubic')
        self.psi = interp1d(x_sel, psi_sel, kind='cubic')
        
    def integrate(self, fpguesses, xmin, xmax, avoidfps=0.1, pert=0.01, M=100):
        self.findfps(fpguesses)
        self.findpotential(xmin, xmax, pert, M)
        self.findprefactor(avoidfps)
        
    def lams(self, eps):
        phiminus = -quad(self.dphi, self.fps[0], self.fps[1])[0]
        phiplus = quad(self.dphi, self.fps[1], self.fps[2])[0]
        psiminus = -quad(self.dpsi, self.fps[0], self.fps[1])[0]
        psiplus = quad(self.dpsi, self.fps[1], self.fps[2])[0]   
        lamplus = 0.5*self.lin_at_fps[1]/np.pi*np.sqrt(-self.d2phi_at_fps[2]/self.d2phi_at_fps[1])* \
                np.exp(psiplus + phiplus/eps)
        lamminus = 0.5*self.lin_at_fps[1]/np.pi*np.sqrt(-self.d2phi_at_fps[0]/self.d2phi_at_fps[1])* \
                np.exp(psiminus + phiminus/eps)
        return lamminus, lamplus

    def weights(self, eps, t=None, xinit=None):
        lamminus, lamplus = self.lams(eps)
        omegaminusinf = lamplus/(lamplus + lamminus)
        omegaplusinf = lamminus/(lamplus + lamminus)
        if t is None or xinit is None:
            return omegaminusinf, omegaplusinf
        omegaminus0 = float(xinit < self.fps[1])
        omegaplus0 = float(xinit > self.fps[1])
        omegaminus = omegaminusinf + (omegaminus0 - omegaminusinf)*np.exp(-(lamplus+lamminus)*t)
        omegaplus = omegaplusinf + (omegaplus0 - omegaplusinf)*np.exp(-(lamplus+lamminus)*t)
        return omegaminus, omegaplus
        
        
        
#if __name__=="__main__":
#    pass        

#if __name__=="__main__":
#    model = SlwTrns()
#    
#    """Define fixed point guesses"""
#    fp_guesses = [2., 6., 10.]
#    xmax = 14
#    eps = np.array([0.06,0.04,0.03])
#    
#    """Uncomment this to run the bursty case"""
#    #initialise(0.5 + 2. * (np.arange(21) >= 6), 4)
#    #eps = np.array([0.03,0.02,0.0125])
#    
#    """Uncomment this to get monostable"""
#    #fp_guesses = [6.]
#    #xmax = 11.
#    #initialise(2. + 8. * (np.arange(21) < 6), 1)
#    #initialise(0.5 + 2. * (np.arange(21) < 6), 4)
#    
#    """Calculate fixed points and the value of Phi''(x) at them"""
#
#        
#    """Values of Phi' and Phi'' on a fine mesh of x values"""    
#    x, dphi, d2phi = get_hetero(model.oderhs, fps, d2phi_at_fps, xmin=1, xmax=xmax)  
#
#    """Values of psi'"""
#    dpsi =np.ndarray(shape=x.shape)
#    for i in range(len(x)):
#        dpsi[i] = model.get_dpsi(x[i], dphi[i], d2phi[i])
#    
#    """select those x's who are far enough from the fixed pts"""
#    sel = np.ones(len(x), dtype=bool)
#    for fp in fps:
#        sel *= np.abs(x - fp) > 0.1
#        
#    """Plotting"""    
#    fig, ax = plt.subplots(6,1,figsize=(4,3*6))
#    ax[0].plot(x, dphi)
#    ax[0].set_ylabel("$\\Phi'(x)$")
#    ax[1].plot(x, d2phi)
#    ax[1].set_ylabel("$\\Phi''(x)$")
#    ax[2].plot(x, dpsi)
#    ax[2].set_ylabel("$\\psi'(x)$ (uncorrected)")
#    ax[3].plot(x[sel],dpsi[sel])
#    ax[3].set_ylabel("$\\psi'(x)$ (corrected)")
#    
#    if len(fps) >= 3:
#        dphi_spline = interp1d(x, dphi, kind='cubic')
#        dpsi_spline = interp1d(x[sel], dpsi[sel], kind='cubic')
#        phiminus = -quad(dphi_spline, fps[0], fps[1])[0]
#        phiplus = quad(dphi_spline, fps[1], fps[2])[0]
#        phimax =  quad(dphi_spline, fps[1], xmax)[0]
#        psiminus = -quad(dpsi_spline, fps[0], fps[1])[0]
#        psiplus = quad(dpsi_spline, fps[1], fps[2])[0]    
#        #wkbdata = {'f': f, 'b': b, 'xmax': xmax, 'fps': fps, 'lin_at_fps': lin_at_fps, 'd2phi_at_fps': d2phi_at_fps,
#        #           'phiminus': phiminus, 'phiplus': phiplus, 'phimax': phimax, 'psiminus': psiminus, 'psiplus': psiplus}
#        omegaratio = np.sqrt(d2phi_at_fps[2]/d2phi_at_fps[0])*np.exp(-psiminus + psiplus - phiminus/eps + phiplus/eps)
#        omegaminus = omegaratio/(1. + omegaratio)
#        omegaplus = 1./(1. + omegaratio)
#        x = np.linspace(0, xmax, 201)
#        for i in range(len(eps)):
#            pmfx = omegaminus[i]*norm.pdf(x, loc=fps[0], scale=np.sqrt(eps[i]/d2phi_at_fps[0])) + \
#                omegaplus[i]*norm.pdf(x, loc=fps[2], scale=np.sqrt(eps[i]/d2phi_at_fps[2]))
#            ax[4].plot(x,pmfx)
#            ax[4].set_ylabel("Gaussian mixture")
#            pmfs = omegaminus[i]*model.rho(fps[0]) + omegaplus[i]*model.rho(fps[2])
#            ax[5].plot(np.arange(len(pmfs)), pmfs)
#            ax[5].set_ylabel("Poisson mixture")
#    
#    plt.tight_layout()
#    plt.show()