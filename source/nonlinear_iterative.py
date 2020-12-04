'''
========================================================================================================================
# Author:       Jordan Leiker
# GTid:         903453031
# Class:        MATH6644 - Iterative Methods for Solving Systems of Equations
# Date:         03/21/2020
========================================================================================================================
'''

import numpy as np
import scipy.linalg as sp

''' Basic Newtons Method
 Arguments:
 func_F            => Function handle for non-linear system: should return numpy array
 func_Jacobian     => Function Handle for Jacobian: should return numpy array
 func_solveMethod  => Function Handle for solver method, can be CG, inverse, etc.
                      Note: This only accepts a A,b in Ax=b. Thus PCG cannot be used in this version
 x                 => starting point: should be numpy array
 tau               =>  [tol for relative error, tol for abs error]
 printIterates     => Print each successive solution every iteration'''
def newtonsMethod(func_F, func_Jacobian, func_solveMethod, x, tau, maxIters, printIterates=False):
    # Intialize
    f_x = func_F(x)
    r_0 = np.linalg.norm(f_x)
    err = r_0
    err_vec = [err]
    numIters = 0

    while (err >= (tau[0]*r_0 + tau[1])) and (numIters < maxIters):
        # Print Iterates (for debug)
        if printIterates:
            print("Iteration " + "{}".format(numIters))
            print(x)

        Jacobian = func_Jacobian(x)
        s, _ = func_solveMethod(A=Jacobian, b=-f_x)
        x = x + s
        f_x = func_F(x)
        numIters += 1
        err = np.linalg.norm(f_x)
        err_vec.append(err)
        # the error grew, so break
        if (err_vec[numIters] - err_vec[numIters-1]) > 0:
            break

    return x, numIters, err_vec


'''Basic Chord Method
 Arguments:
 func_F            => Function handle for non-linear system: should return numpy array
 func_Jacobian     => Function Handle for Jacobian: should return numpy array
 x                 => starting point: should be numpy array
 tau               =>  [tol for relative error, tol for abs error]
 printIterates     => Print each successive solution every iteration'''
def chordMethod(func_F, func_Jacobian, x, tau, printIterates=False):
    # Intialize
    f_x = func_F(x)
    r_0 = np.linalg.norm(f_x)
    numIters = 0

    # Calculate Initial Jacobian and LU Factorize
    Jacobian = func_Jacobian(x)
    if Jacobian.shape[0] > 1:
        (LU,piv) = sp.lu_factor(Jacobian)

    while np.linalg.norm(f_x) >= (tau[0]*r_0 + tau[1]):
        # Print Iterates (for debug)
        if printIterates:
            print("Iteration " + "{}".format(numIters))
            print(x)

        if Jacobian.shape[0] > 1:
            s = sp.lu_solve((LU,piv), -f_x)
        else:
            # Use inverse because I'm treating everything (even 1D) as arrays
            # This is effectively a divide and will never, ever, be used on matrices
            s = np.linalg.inv(Jacobian)*(-f_x)
        x = x + s
        f_x = func_F(x)
        numIters += 1

    return x, numIters


''' Basic Secant Method (only 1D)
 Arguments:
 func_F            => Function handle for non-linear system
 xn1               => starting point at time t = -1 (secant method requires two points to start)
 x0                => starting point at time t = 0
 tau               =>  [tol for relative error, tol for abs error]
 printIterates     => Print each successive solution every iteration'''
def secantMethod(func_F, x0, xn1, tau, printIterates=False):
    # Intialize
    numIters = 0

    # Calculate initial value
    x_old = xn1
    x = x0
    f_old = func_F(x_old)
    f = func_F(x)
    r_0 = np.abs(f)

    while np.linalg.norm(f) >= (tau[0]*r_0 + tau[1]):
        # Print Iterates (for debug)
        if printIterates:
            print("Iteration " + "{}".format(numIters))
            print(x)

        a = (f - f_old) / (x - x_old)
        x_old = x
        x = x - f * (1 / a)
        f_old = f
        f = func_F(x)

        numIters += 1

    return x, numIters


# Note: This is the dirder function from C.T. Kelley
# "Iterative Methods" for Linear and Non-Linear Equations"
# but ported to Python from Matlab
'''Finite difference directional derivative
 Approximate f'(x) w

 C. T. Kelley, November 25, 1993
 This code comes with no guarantee or warranty of any kind.

 function z = dirder(x,w,f,f0)
 inputs:
           x, w = point and direction
           f = function
           f0 = f(x), in nonlinear iterations
                f(x) has usually been computed
                before the call to dirder'''
def dirder(x, w, f, f0):
    # Hardwired difference increment.
    epsnew = 1.e-7
    n = x.shape[0]

    # scale the step
    if np.linalg.norm(w) == 0:
        z = np.zeros((n, 1))
        return z

    epsnew = epsnew / np.linalg.norm(w)
    if np.linalg.norm(x) > 0:
        epsnew = epsnew * np.linalg.norm(x)

    # del and f1 could share the same space if storage
    # is more important than clarity
    delx = x + epsnew * w
    f1 = f(delx)
    z = (f1 - f0) / epsnew
    return z


# Note: This is the diffjac function from C.T. Kelley
# "Iterative Methods" for Linear and Non-Linear Equations"
# but ported to Python from Matlab
'''compute a forward difference Jacobian f'(x)

 uses dirder.m to compute the columns

 C. T. Kelley, November 25, 1993
 This code comes with no guarantee or warranty of any kind.

 inputs:
         x, f = point and function
        f0   = f(x), preevaluated'''
def diffjac(x, f, f0):
    n = x.shape[0];
    jac = np.zeros((n, n))

    for j in range(n):
        zz = np.zeros((n, 1))
        zz[j] = 1
        jac[:, j] = np.ravel(dirder(x, zz, f, f0))

    return jac


''' Flexible Newtons Method - Choice of m selects Newtons Method, Chord method, or Shamanskii Method. In all
 cases the Jacobian is calculated using a finite difference method jacobian. (from T. Kelley)
 Note that only LU solving techniques are used here. This function does not accept alternative solving methods 
 for the inner loop.
 
 Arguments:
 func_F            => Function handle for non-linear system: should return numpy array
 x                 => starting point: should be numpy array
 tau               =>  [tol for relative error, tol for abs error]
 m                 => Number of iterations for Jacobian (m=1 (Newton), m=infinity (Chord), 1<m<inf (Sham)
                    default Newtons Method (m=1)
 printIterates     => Print each successive solution every iteration'''
def nonlinearMethods(func_F, x, tau, maxIters, m=1, printIterates=False):
    # Intialize
    f_x = func_F(x)
    r_0 = np.linalg.norm(f_x)
    err = r_0
    err_vec = [err]
    numIters = 0

    while (err > (tau[0]*r_0 + tau[1])) and (numIters < maxIters):
        Jacobian = diffjac(x,func_F,f_x)
        (LU, piv) = sp.lu_factor(Jacobian)
        for j in range(m):
            # Print Iterates (for debug)
            if printIterates:
                print("Iteration " + "{}".format(numIters))
                print(x)

            s = sp.lu_solve((LU,piv), -f_x)
            x = x + s
            f_x = func_F(x)
            err = np.linalg.norm(f_x)
            err_vec.append(err)
            numIters += 1

            if np.linalg.norm(f_x) <= (tau[0]*r_0 + tau[1]):
                break

    return x, numIters, err_vec

'''Generic Fixed Point Method

 Arguments:
 func_F            => Function handle for non-linear system: should return numpy array
 x                 => starting point: should be numpy array
 tau               =>  [tol for relative error, tol for abs error]
 printIterates     => Print each successive solution every iteration'''
def fixedPointMethod(func_F, x, tau, maxIters, printIterates=False):
    # Intialize
    f_x = func_F(x)
    r_0 = np.linalg.norm(f_x)
    err = r_0
    err_vec = [err]
    numIters = 0

    while (err > (tau[0]*r_0 + tau[1])) and (numIters < maxIters):
        # Print Iterates (for debug)
        if printIterates:
            print("Iteration " + "{}".format(numIters))
            print(x)

        x = x - f_x
        f_x = func_F(x)
        err = np.linalg.norm(f_x)
        err_vec.append(err)
        numIters += 1

    return x, numIters, err_vec