import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4
import time
from conj_grad import helper_PCG_direct

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")
psi4.set_options({"basis": "cc-pVDZ",
                  "scf_type": "df",
                  "cphf_tasks": ['polarizability']})

scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)


numpy_memory = 2
C = scf_wfn.Ca()
Co = scf_wfn.Ca_subset("AO", "OCC")
Cv = scf_wfn.Ca_subset("AO", "VIR")
epsilon = np.asarray(scf_wfn.epsilon_a())

nbf = scf_wfn.nmo()
nocc = scf_wfn.nalpha()
nvir = nbf - nocc

# Integral generation from Psi4's MintsHelper
mints = psi4.core.MintsHelper(scf_wfn.basisset())
S = np.asarray(mints.ao_overlap())

# Transformation of perturbation tensors from AO to MO basis
nCo = np.asarray(Co)
nCv = np.asarray(Cv)
tmp_dipoles = mints.so_dipole()
dipoles_xyz = []
for num in range(3):
    Fso = np.asarray(tmp_dipoles[num])
    Fia = (nCo.T).dot(Fso).dot(nCv)
    Fia *= -2
    dipoles_xyz.append(Fia)

# Computation of  electronic hessian

print('\nForming hessian...')
docc = np.diag(np.ones(nocc))
dvir = np.diag(np.ones(nvir))
eps_diag = epsilon[nocc:].reshape(-1, 1) - epsilon[:nocc]

MO = np.asarray(mints.mo_eri(Co, C, C, C))
H = np.einsum('ai,ij,ab->iajb', eps_diag, docc, dvir)
H += 4 * MO[:, nocc:, :nocc, nocc:]
H -= MO[:, nocc:, :nocc, nocc:].swapaxes(0, 2)
H -= MO[:, :nocc, nocc:, nocc:].swapaxes(1, 2)

# Using conjugate solver to solve the CPHF equations:

x = {}

start = time.time()	
for numx in range(3):
    x[numx] = helper_PCG_direct(H.reshape(nocc*nvir,-1), dipoles_xyz[numx].reshape(nocc*nvir))
stop = time.time()
print("total time for my CG:{}".format(stop - start))

# Compute 3x3 polarizability tensor
polar = np.empty((3, 3))
polar1 = np.empty((3, 3))
for numx in range(3):
    for numf in range(3):
        polar[numx, numf] = -1 * np.dot(x[numx], dipoles_xyz[numf].reshape(nocc*nvir))

print('\nCPHF Dipole Polarizability: ( CG Direct)')
print(np.around(polar, 5))
