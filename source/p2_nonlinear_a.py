'''
========================================================================================================================
Author:       Jordan Leiker
GTid:         903453031
Class:        MATH6644 - Iterative Methods for Solving Systems of Equations
Date:         04/21/2020

Title:        Project 2 (Non-Linear)
Description:  (a) Solve Chandrasekhar H-Equations using Newton, Chord, Shammanskii, and Fixed Point methods. Anaylze.
========================================================================================================================
'''

import numpy as np
import nonlinear_iterative as nl
import matplotlib.pyplot as plt
import time
import sys
import pandas

desiredWidth = 400
pandas.set_option('display.width', desiredWidth)
pandas.set_option('display.max_columns', 10)
np.set_printoptions(linewidth=desiredWidth)


'''Function to evaluate the discrete Chandrasekhar H Equations
 i.e. 
 Fi(x) = xi - (1 - (c/2N) sum_N( mi*xj / mi+mj ) )^-1 = 0
 where c is in (0,1) and m = (i - 1/2)/N for 1 <= i <= N'''
def chandrasekharHEq(x, c):
    n = x.shape[0]
    # Work in single dimension
    fx = np.zeros(n)
    x = np.ravel(x)
    m = (np.arange(1,n+1)-0.5)/n
    for i in range(n):
        sum_j = np.sum((m[i] * x)/(m[i] + m))
        fx[i] = x[i] - (1-(c/(2*n))*sum_j)**(-1)
    return np.atleast_2d(fx).T # return "2d", i.e. n x 1


'''Project 2 (a)'''
if __name__ == '__main__':
    print("-----------------------------------------------------------------------------------------------------------")
    print("------------------------------    Project 2 - Non-Linear (a)     ------------------------------------------")
    print("-----------------------------------------------------------------------------------------------------------")
    print("\n")
    PRINT_ENABLE = False
    PLOT_ENABLE = True
    # Environment Wide Variables (all of these given on the assignment)
    # ------------------------------------------------------------------------------------------------------------------
    relError = 1e-6
    absError = 1e-6
    tau = (relError, absError)
    N = 200
    c = 0.9
    x0 = np.ones((N,1))
    func_chandra = lambda x: chandrasekharHEq(x,c) # lambda so I can pass the function handle but set c here

    # Store results from solvers
    methods = ("Newtons Method", "Shamanskii Method", "Chord Method", "Fixed Point Method")
    numMethods = len(methods)
    x = np.zeros((N,numMethods))
    numIters = np.zeros((1,numMethods))
    times = np.zeros((1,numMethods))
    norms = np.zeros((1,numMethods))
    err_vecs = []


    # Start Solving
    # ------------------------------------------------------------------------------------------------------------------
    print("--------------------------------    Newtons Method - Find Roots      --------------------------------------")
    start = time.time()
    xtemp, numIters[:,0], err_vec = nl.nonlinearMethods(func_chandra,x0,tau,maxIters=20,m=1,printIterates=False)
    end = time.time()
    x[:,0] = np.ravel(xtemp)
    norms[:,0] = np.linalg.norm(func_chandra(x[:,0]))
    err_vecs.append(err_vec)
    times[:, 0] = end - start

    print("---------------------------    Shamanskii Method (m=2) - Find Roots      ----------------------------------")
    start = time.time()
    xtemp, numIters[:,1], err_vec = nl.nonlinearMethods(func_chandra,x0,tau,maxIters=20,m=2,printIterates=False)
    end = time.time()
    x[:,1] = np.ravel(xtemp)
    norms[:, 1] = np.linalg.norm(func_chandra(x[:, 1]))
    err_vecs.append(err_vec)
    times[:, 1] = end - start

    print("----------------------------------    Chord Method - Find Roots      --------------------------------------")
    # Note: For Chord Method m = infinity. Using sys.maxsize (the max integer size) here because we will never reach
    # this so it doesn't matter that it's actually finite, and m is used as a loop "range" and thus needs to be an int.
    start = time.time()
    xtemp, numIters[:,2], err_vec = nl.nonlinearMethods(func_chandra,x0,tau,maxIters=20,m=sys.maxsize,printIterates=False)
    end = time.time()
    x[:,2] = np.ravel(xtemp)
    norms[:, 2] = np.linalg.norm(func_chandra(x[:, 2]))
    err_vecs.append(err_vec)
    times[:,2] = end-start

    print("------------------------------    Fixed-point Method - Find Roots      ------------------------------------")
    start = time.time()
    xtemp, numIters[:,3], err_vec = nl.fixedPointMethod(func_chandra,x0,tau,maxIters=100,printIterates=False)
    end = time.time()
    x[:,3] = np.ravel(xtemp)
    norms[:, 3] = np.linalg.norm(func_chandra(x[:, 3]))
    err_vecs.append(err_vec)
    times[:,3] = end-start


    # Analyze Results
    # ------------------------------------------------------------------------------------------------------------------
    print("-----------------------------------------    Results      -------------------------------------------------")
    data = np.vstack((numIters[0,:], times[0,:], norms[0,:])).T
    print(pandas.DataFrame(data,methods,columns=['Iters','Times', 'Norms']))
    print("\n")

    if PRINT_ENABLE:
        for i in range(numMethods):
            print(methods[i])
            # Calculate function value evaluation at solution (then calc the norm, should be 0 vec and 0)
            should_be_zero = func_chandra(x[:,i])
            print("Solution, x =")
            print(x[:,i])
            print("Error Vector = ")
            print(err_vecs[i])
            print("F(x) = ")
            print(should_be_zero.T)
            print("||F(x)|| =", np.linalg.norm(should_be_zero))
            print("\n")

    # Plot Error
    if PLOT_ENABLE:
        # Num Iterations and Time to Converge Plots
        fig, ax = plt.subplots()
        ax.set_ylabel('# Iterations', color='red')
        ln1 = ax.plot(methods, numIters[0,:], label="# Iterations", color='red', marker='o', linestyle='--')
        ax.tick_params(axis='y',labelcolor='red')
        ax_2 = ax.twinx()
        ln2 = ax_2.plot(methods, times[0, :], label="Run Time", color='blue', marker='o', linestyle='--')
        ax_2.set_ylabel('Time', color='blue')
        ax_2.tick_params(axis='y', labelcolor='blue')
        ax.set(title='Iterations and Run-Times per Method')
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=5)

        # Error Plots
        stop_condition = tau[0]*np.linalg.norm(func_chandra(x0)) + tau[1]

        fig, ax = plt.subplots()
        ax.hlines(stop_condition, 0, np.max(numIters), color='red', linestyle='--')
        ax.plot(err_vecs[0], label="Newton", color='green', marker="x")
        ax.plot(err_vecs[1], label="Shamanskii (m=2)", color='purple', marker="x")
        ax.plot(err_vecs[2], label="Chord", color='orange', marker="x")
        ax.plot(err_vecs[3], label="Fixed Point", color='blue', marker="x")
        ax.set(xlabel='Iteration', ylabel='Error', title='Error Reduction per Iteration')
        ax.legend()

        fig, ax = plt.subplots()
        ax.hlines(np.log10(stop_condition), 0, np.max(numIters), color='red', linestyle='--')
        ax.plot(np.log10(err_vecs[0]), label="Newton", color='green', marker="x")
        ax.plot(np.log10(err_vecs[1]), label="Shamanskii (m=2)", color='purple', marker="x")
        ax.plot(np.log10(err_vecs[2]), label="Chord", color='orange', marker="x")
        ax.plot(np.log10(err_vecs[3]), label="Fixed Point", color='blue', marker="x")
        ax.set(xlabel='Iteration', ylabel='Error', title='Error Reduction per Iteration, Log10')
        ax.legend()

        plt.show()