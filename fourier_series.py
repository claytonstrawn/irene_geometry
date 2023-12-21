import scipy
from scipy.integrate import quad
from scipy.optimize import brentq
from irene_geometry.utils import check_segments_intersect
from irene_geometry.constants import VELOCITY
import numpy as np

def fourier_series(fs,amplitudes,phases,R,max_n = 10,ret_funcs = False,velocity = 'default'):
    if velocity == 'default':
        v = VELOCITY
    else:
        v = velocity
    X = v/np.gcd.reduce(fs)
    def y(x):
        out = 0
        for i,f in enumerate(fs):
            X_f = v/f
            k = 2*np.pi/X_f
            A = amplitudes[i]
            phi = phases[i]
            out += A*np.sin(k*x+phi)
        return out
    def yprime(x):
        out = 0
        for i,f in enumerate(fs):
            X_f = v/f
            k = 2*np.pi/X_f
            A = amplitudes[i]
            phi = phases[i]
            out += A*k*np.cos(k*x+phi)
        return out
    def yprimeprime(x):
        out = 0
        for i,f in enumerate(fs):
            X_f = v/f
            k = 2*np.pi/X_f
            A = amplitudes[i]
            phi = phases[i]
            out += -A*k**2*np.sin(k*x+phi)
        return out

    def norm(eta):
        return np.sqrt(yprime(eta)**2+1)
    def gamma_x(eta):
        return eta - R*yprime(eta)/norm(eta)
    def gamma_y(eta):
        return y(eta) + R/norm(eta)
    def gamma_x_prime(eta):
        return 1 - R*yprimeprime(eta)/norm(eta)**3
    crit_points,crit_signs = find_crit_points(X,v,fs,yprime)
    if crit_signs[0] == -1:
        crit_points = np.concatenate([crit_points[1:], crit_points[:1]],  axis=0)
        crit_signs = np.concatenate([crit_signs[1:], crit_signs[:1]],  axis=0)
    crit_points = np.concatenate([crit_points, crit_points[:1]+X],  axis=0)
    crit_signs = np.concatenate([crit_signs, crit_signs[:1]],  axis=0)
    all_bounds = []
    last_crit = crit_points[0]
    for i,crit in enumerate(crit_points):
        type_ii = check_type_ii(crit,gamma_x_prime)
        if type_ii:
            left_edge,right_edge = get_bounds_type_ii(i,crit_points,gamma_x,gamma_y,gamma_x_prime)
            if left_edge is None:
                print('intersection not found, ball might be too big for frequencies used here')
                print('returning gamma_x,gamma_y for exploration')
                return X,_,_,y,gamma_x,gamma_y,_
        else:
            left_edge,right_edge = crit,crit
        all_bounds.append((last_crit,left_edge))
        last_crit = right_edge
    all_bounds.append((last_crit,crit_points[0]+X))
    C_coeffs = []
    S_coeffs = []
    for k in range(max_n+1):
        C_k,S_k = kth_fourier_coeff(k,X,gamma_x,gamma_y,gamma_x_prime,all_bounds)
        C_coeffs.append(C_k)
        S_coeffs.append(S_k)
    if ret_funcs:
        return X,C_coeffs,S_coeffs,y,gamma_x,gamma_y,gamma_x_prime
    else:
        return X,C_coeffs,S_coeffs

def get_bounds_type_ii(i,crit_points,gamma_x,gamma_y,gamma_x_prime,starting_n_etas = 1000):
    crit_to_replace = crit_points[i]
    left_crit = crit_points[i-1]
    right_crit = crit_points[i+1]
    start_point = brentq(gamma_x_prime,left_crit,crit_to_replace)
    end_point = brentq(gamma_x_prime,crit_to_replace,right_crit)
    u_etas = np.linspace(left_crit,start_point,starting_n_etas)
    u_xs,u_ys = gamma_x(u_etas),gamma_y(u_etas)
    v_etas = np.linspace(end_point,right_crit,starting_n_etas)
    v_xs,v_ys = gamma_x(v_etas),gamma_y(v_etas)
    for i in range(len(u_xs)-1):
        ux1,uy1 = u_xs[i],u_ys[i]
        ux2,uy2 = u_xs[i+1],u_ys[i+1]
        for j in range(len(v_xs)-1):
            vx1,vy1 = v_xs[j],v_ys[j]
            vx2,vy2 = v_xs[j+1],v_ys[j+1]
            x_int = check_segments_intersect(ux1,ux2,uy1,uy2,vx1,vx2,vy1,vy2)
            if x_int is not None:
                def gamma_x_diff(eta):
                    return gamma_x(eta) - x_int
                left_edge = brentq(gamma_x_diff,left_crit,start_point)
                right_edge = brentq(gamma_x_diff,end_point,right_crit)
                return left_edge,right_edge
    return None,None
    
def kth_fourier_coeff(k,X,gamma_x,gamma_y,gamma_x_prime,all_bounds):
    if k == 0:
        prefactor = 1
    else:
        prefactor = 2
    def C_integrand(eta):
        return prefactor/X*gamma_y(eta)*np.cos(2*np.pi*k/X*gamma_x(eta))*gamma_x_prime(eta)
    def S_integrand(eta):
        return prefactor/X*gamma_y(eta)*np.sin(2*np.pi*k/X*gamma_x(eta))*gamma_x_prime(eta)
    C_k = 0
    S_k = 0
    for bound in all_bounds:
        C_k += quad(C_integrand,bound[0],bound[1])[0]
        S_k += quad(S_integrand,bound[0],bound[1])[0]
    return C_k,S_k

def check_type_ii(crit,gamma_x_prime):
    if gamma_x_prime(crit)<0:
        return True
    else:
        return False
            
def find_crit_points(X,v,fs,yprime,starting_n_xs = 100):
    highest_freq = np.max(fs)
    max_crit_points = 2*X/(v/highest_freq)
    approx_crits = []
    n_xs = starting_n_xs
    while len(approx_crits)<max_crit_points:
        xs = np.linspace(0,X,n_xs)
        dys = yprime(xs)
        loc_of_flips = (np.diff(np.sign(dys)) != 0)
        loc_of_flips = np.concatenate([loc_of_flips, loc_of_flips[:1]],  axis=0)
        if len(approx_crits) == len(xs[loc_of_flips]):
            break
        approx_crits = xs[loc_of_flips]
        n_xs*=10
    real_crits = approx_crits*0.0
    real_crit_signs = approx_crits*0.0
    left_bound = 0
    for i,approx_crit in enumerate(approx_crits):
        if i == len(approx_crits)-1:
            right_bound = X
        else:
            right_bound = (approx_crit+approx_crits[i+1])/2
        real_crits[i] = brentq(yprime,left_bound,right_bound)
        real_crit_signs[i] = 1 if yprime(left_bound) > 0 else -1
        left_bound = right_bound
    return real_crits,real_crit_signs

def fourier_approx(X,C_coeffs,S_coeffs):
    def g(x):
        out = 0.0*x
        for n,Cn in enumerate(C_coeffs):
            out+=Cn*np.cos(2*np.pi*n/X*x)
        for n,Sn in enumerate(S_coeffs):
            out+=Sn*np.sin(2*np.pi*n/X*x)
        return out
    return g