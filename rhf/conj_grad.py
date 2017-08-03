import numpy as np
"""
Contains the conjugate gradient helper function
"""
def helper_PCG_direct(A, b, tol=1e-10, max_iter=None, x0=None, M=None):
    """
    Solves a linear system of equations Ax = b using a preconditioned conjugate 
    gradient method within a user-defined tolerance
    

    Parameters
    ----------
    A : LHS Matrix (2-dimensional numpy array)
    b : RHS vector (1-dimensional numpy array)
    tol : User-defined tolerance for residual convergence (Default = 1.e-10)
    max_iter : User-defined maximum number of iterations for convergence (Default = 2 * len(b))
    x0 : User-specified guess vector (Default = zero-array)
    M : Preconditioner Matrix (2-dimensional numpy array)  (Default = Identity Matrix)  

    Returns
    -------
    x : solution vector of Ax = b

    Notes
    -----
    
    Examples
    --------

    x = helper_PCG_direct(A, b)
    ...

    """

    if max_iter is None:
        max_iter = 2*len(b)
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0
    if M is None:
        M = np.diag(A)

    r = b - np.dot(A,x)
    z = r/M
    n = len(b)
    p = z
    res_k_norm = np.dot(r, z)
    print("Starting Conjugate Gradient Iterations...\n")
    for iteration in range(2*n):
        Ap = np.dot(A, p)
        alpha = res_k_norm / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        z = r/M
        res_kplus1_norm = np.dot(r, z)
        beta = res_kplus1_norm / res_k_norm
        res_k_norm = res_kplus1_norm
        rms = np.sqrt(np.sum(r**2) / len(r))
        print('CG Iteration %3d: RMS = %3.8f' % (iteration, rms))
        if (res_kplus1_norm < tol) or (iteration > max_iter):
            print('\nConverged!!\n')
            break
        p = beta * p + z
    return x

