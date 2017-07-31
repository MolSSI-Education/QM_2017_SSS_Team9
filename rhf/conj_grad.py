import numpy as np
import time


def conj_grad(A, b):
    n = len(b)
    x = np.ones(n) 
    r = np.dot(A, x) - b
    p = - r
    res_k_norm = np.dot(r, r)
    for i in range(2*n):
        Ap = np.dot(A, p)
        alpha = res_k_norm / np.dot(p, Ap)
        x += alpha * p
        r += alpha * Ap
        res_kplus1_norm = np.dot(r, r)
        beta = res_kplus1_norm / res_k_norm
        res_k_norm = res_kplus1_norm
        if res_kplus1_norm < 1e-14:
            print('Total iterations:', i)
            break
        p = beta * p - r
    return x

if __name__ == '__main__':
    n = 100
    A = np.random.normal(size=[n, n])
    A = np.dot(A.T, A)
    b = np.ones(n)
    xo = np.zeros(n)
    x = conj_grad(A, b, xo)
    x1 = np.linalg.solve(A, b)
    print(" Solution matches with numpy's cg solver: %s" % np.allclose(x, x1, 1e-9))

