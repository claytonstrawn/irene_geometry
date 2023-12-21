import scipy
from scipy.integrate import quad
from scipy.optimize import brentq
from irene_geometry.constants import VELOCITY
import numpy as np

def fourier_series_pure_tone(f,amplitude,R,max_n = 10,ret_funcs = False,velocity = 'default'):
    if velocity == 'default':
        X = VELOCITY/f
    else:
        X = velocity/f
    A = amplitude
    from numpy import sin,cos,sqrt
    k = 2*np.pi/X
    
    def gamma_x(eta):
        return eta - R*A*sin(k*eta)/sqrt(A**2*sin(k*eta)**2+1)
    def gamma_y(eta):
        return -A/k*cos(k*eta)+R/sqrt(A**2*sin(k*eta)**2+1)
    def gamma_x_prime(eta):
        return 1 - R*A*k*cos(k*eta)/(A**2*sin(k*eta)**2+1)**(3/2)
    
    eta0 = find_eta0(X,gamma_x,gamma_x_prime)
    coeffs = []
    for n in range(max_n+1):
        coeffs.append(puretone_nth_fourier_coeff(n,X,gamma_x,gamma_y,gamma_x_prime,eta0))
    if ret_funcs:
        return X,coeffs,gamma_x,gamma_y,gamma_x_prime
    else:
        return X,coeffs
    
def find_eta0(X,gamma_x,gamma_x_prime):
    if gamma_x_prime(0) >= 0:
        return 0
    crit = brentq(gamma_x_prime,-X/2,0)
    eta0 = brentq(gamma_x,-X/2,crit)
    return eta0

def puretone_nth_fourier_coeff(n,X,gamma_x,gamma_y,gamma_x_prime,eta0):
    if n == 0:
        prefactor = 2
    else:
        prefactor = 4
    def integrand(eta):
        return prefactor/X*gamma_y(eta)*np.cos(2*np.pi*n/X*gamma_x(eta))*gamma_x_prime(eta)
    return quad(integrand,-X/2,eta0)[0]

def puretone_fourier_approx(X,coeffs):
    def g(x):
        out = 0.0*x
        for n,Cn in enumerate(coeffs):
            out+=Cn*np.cos(2*np.pi*n/X*x)
        return out
    return g