import sys
sys.path.append('/Users/boyi/molssi_sss/Gits/QM_2017_SSS_Team9_new-/JK_binding/build')
import numpy as np
import psi4
import jk_binding    
import time

# Make sure we get the same random array
np.random.seed(0)

# A hydrogen molecule
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")

# Build a ERI tensor
basis = psi4.core.BasisSet.build(mol, target="cc-pvdz")
mints = psi4.core.MintsHelper(basis)
I = np.array(mints.ao_eri())

# Symmetric random density
nbf = I.shape[0]
D = np.random.rand(nbf, nbf)
D = (D + D.T) / 2

# Reference
t1 = time.time()
J_ref = np.einsum("pqrs,rs->pq", I, D)
t2 = time.time()
print("J build time:", t2-t1)
K_ref = np.einsum("prqs,rs->pq", I, D)
I_ = np.zeros_like(D)
def index(i, j):
    if i < j:
        return j*(j + 1)/2
    else:
        return i*(i+1)/2
for i in range(nbf):
    for j in range(nbf):
        ij = index(i,j)
        for k in range(nbf):    
            for l in range(nbf):
                kl = index(k,l)
                I_[ij,kl] = I[i,j,k,l]
    # Your implementation
t1 = time.time()
J = jk_binding.form_j(I.reshape(nbf*nbf, nbf*nbf),D.reshape(nbf*nbf))
J = J.reshape(nbf, nbf)
t2 = time.time()
print("J build time:", t2-t1)

#K = np.random.rand(nbf, nbf)
t1 = time.time()
K = jk_binding.form_k(I,D)
t2 = time.time()
print("K build time:", t2-t1)
# Make sure your implementation is correct
print("J is correct: %s" % np.allclose(J, J_ref))
print("K is correct: %s" % np.allclose(K, K_ref))
