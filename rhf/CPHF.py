import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4
import time

def scf_response(scf_wfn,options):
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
    for numx in range(3):
        n = len(dipoles_xyz[numx])
        x[numx] = np.ones_like(dipoles_xyz[numx])
        r = np.einsum('iajb,jb->ia',H,x[numx]) - dipoles_xyz[numx]
        p = - r
        res_k_norm = np.einsum('ia,ia->', r, r)
        for i in range(10*n):
            Ap = np.einsum('iajb,jb->ia',H, p)
            tmp = np.einsum('ia,ia->',p,Ap)
            alpha = res_k_norm/tmp
            x[numx] += alpha * p
            r += alpha * Ap
            res_kplus1_norm = np.einsum('ia,ia->',r,r)
            beta = res_kplus1_norm / res_k_norm
            res_k_norm = res_kplus1_norm
            if res_kplus1_norm < 1e-7:
                print('Total iterations:', i)
                break
            p = beta * p - r
    
    
    # Compute 3x3 polarizability tensor
    polar = np.empty((3, 3))
    for numx in range(3):
        for numf in range(3):
            polar[numx, numf] = -1 * np.einsum('ia,ia->', x[numx], dipoles_xyz[numf])
    
    print('\nCPHF Dipole Polarizability:')
    print(np.around(polar, 5))
