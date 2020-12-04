'''
========================================================================================================================
Author:       Jordan Leiker
GTid:         903453031
Class:        MATH6644 - Iterative Methods for Solving Systems of Equations
Date:         04/21/2020

Title:        Project 2 (Non-Linear)
Description:  (b) Use Newtons Method to find as many roots as possible for the given non-linear system
========================================================================================================================
'''

import numpy as np
import scipy.linalg as sp
import nonlinear_iterative as nl

def methodInverse(A,b):
   return np.linalg.inv(A)@b, 0

def luSolve(A,b):
    (LU, piv) = sp.lu_factor(A)
    return sp.lu_solve((LU, piv), b), 0

'''Setup non-linear system (provided on project)
 eq1: xy = z^2 + 1             ===>  0 = z^2 + 1 - xy
 eq2: xyz + y^2 = x^2 + 2      ===>  0 = xyz + y^2 - x^2 - 2
 eq3: exp(x) + z = exp(y) + 3  ===>  0 = exp(x) - exp(y) + z - 3'''
def function(x):
    n = x.shape[0]
    f = np.zeros((n,1))
    f[0,0] = x[2]**2 + 1 - x[0]*x[1]
    f[1,0] = x[0]*x[1]*x[2] + x[1]**2 - x[0]**2 - 2
    f[2,0] = np.exp(x[0]) - np.exp(x[1]) + x[2] - 3
    return f

''' Directly compute Jacobian, because system is small I can compute each partial derivative expression
 directly instead of using an analytic/finite difference jacobian method'''
def jacobian(x):
   n = x.shape[0]
   J = np.zeros((n,n))
   J[0, 0] = -x[1]
   J[0, 1] = -x[0]
   J[0, 2] = 2*x[2]
   J[1, 0] = x[1]*x[2]-2*x[0]
   J[1, 1] = x[0]*x[2]+2*x[1]
   J[1, 2] = x[0]*x[1]
   J[2, 0] = np.exp(x[0])
   J[2, 1] = -np.exp(x[1])
   J[2, 2] = 1
   return J


'''Project 2 (b)'''
if __name__ == '__main__':
    print("-----------------------------------------------------------------------------------------------------------")
    print("------------------------------    Project 2 - Non-Linear (b)     ------------------------------------------")
    print("-----------------------------------------------------------------------------------------------------------")
    # Environment Wide Variables
    # ------------------------------------------------------------------------------------------------------------------
    # Convergence parameters for Newtons Method
    relErrorN = 1e-4
    absErrorN = 1e-4
    tauN = (relErrorN, absErrorN)
    # Convergence parameters for Fixed Point (if used)
    relErrorFP = 1e-6
    absErrorFP = 1e-6
    tauFP = (relErrorFP, absErrorFP)

    # Root Finding
    # ------------------------------------------------------------------------------------------------------------------
    numRootsFound = 0
    loopCount = 0
    stepSize = 0.5
    vol = 10
    xx = np.arange(-vol,vol,stepSize)
    yy = np.arange(-vol,vol,stepSize)
    zz = np.arange(-vol,vol,stepSize)
    xxx, yyy, zzz = np.meshgrid(xx,yy,zz)
    x = np.empty((3,1))

    # while loopCount < xxx.shape[0]*xxx.shape[1]*xxx.shape[2]:
    #     if np.mod(loopCount,50) == 0:
    #         print("LoopCount = " + "{}".format(loopCount))

    # Generate guesses over entire volume
    for i in range(xxx.shape[0]):
        for j in range(xxx.shape[1]):
            for k in range(xxx.shape[2]):
                # Print every 50 to not flood the console but still maintain sanity
                if np.mod(loopCount, 100) == 0:
                    print("LoopCount = " + "{}".format(loopCount) + "/" + "{}".format(xxx.shape[0]**3))

                x0 = np.array([[xxx[i,j,k]],[yyy[i,j,k]],[zzz[i,j,k]]])

                # Use Fixed Point to approach a root from the guess
                # x00, itersFP,_= nl.fixedPointMethod(function, x0, tauFP, maxIters=50, printIterates=False)

                # x=y=z=0 is NOT a solution and breaks things. So don't evaluate it
                if not(np.all(x0 == 0)):
                    # Newtons Method to take it the rest of the way
                    xt, itersN, _ = nl.newtonsMethod(function, jacobian, methodInverse, x0, tauN, maxIters=5)

                    # Check if xt is even a root (we aren't guaranteed to converge if we start too far from a root)
                    if np.allclose(np.ravel(function(xt)),np.zeros(3), relErrorN, absErrorN):

                        # Check if we already found that root
                        if numRootsFound == 0:
                            x = xt
                            numRootsFound += 1
                        else:
                            rootFound = False
                            for i in range(numRootsFound):
                                # cycle through each root and compare to latest result
                                rootFound = rootFound or np.allclose(x[:,i],np.ravel(xt), relErrorN, absErrorN)
                                # We already found the root, break and stop checking
                                if rootFound:
                                    break
                            # The current root does not match any in our list. Add it
                            if not(rootFound):
                                x = np.append(x,xt, axis=1)
                                numRootsFound += 1

                loopCount += 1

    print("Number of Roots Found:\t" + "{}".format(numRootsFound))
    for i in range(numRootsFound):
        print("x["+"{}".format(i)+"] =\t"+"{}".format(x[:,i].T))


