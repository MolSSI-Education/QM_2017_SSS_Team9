"""
MP2 is a post-HF method that improves upon HF by adding electron correlation via
a perturbation method.
"""
import numpy as np

def mp2(molecule, hfe):

    eps = molecule.e

    #Compute (ia|jb)
    temp = np.einsum("pi,pqrs->iqrs", molecule.C, molecule.g)
    temp = np.einsum("rj,iqrs->iqjs", molecule.C, temp)
    temp = np.einsum("qa,iqjs->iajs", molecule.C, temp)
    iajb = np.einsum("sb,iajs->iajb", molecule.C, temp)

    #Compute the opposite spin (OS) and same spin (SS) terms
    e_os = 0.0
    e_ss = 0.0

    for i in range(molecule.nelec):
        for j in range(molecule.nelec):
            for a in range(molecule.nelec, len(molecule.C)):
                for b in range(molecule.nelec, len(molecule.C)):
                    e_os += (iajb[i, a, j, b]*iajb[i, a, j, b])/(eps[i]+eps[j]-eps[a]-eps[b])
                    e_ss += ((iajb[i, a, j, b]-iajb[i, b, j, a])*iajb[i, a, j, b])/(eps[i]+eps[j]-eps[a]-eps[b])

    #Apply a spin-component correction (SCS-MP2)
    e_mp2 = (1./3.)*e_ss+(6./5.)*e_os

    #Compute the total energy
    e_total = e_mp2 + hfe

    return e_total
