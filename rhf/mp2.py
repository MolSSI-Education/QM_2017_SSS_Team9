"""
MP2 Module
"""
import numpy as np
import psi4
import rhf


#What I need:
#Number of occupied orbitals
#Number of virtual orbitals/Total number of orbitals
#MO occupied orbbital energies
#MO virtual orbital energies

#nel = #
#eps = from HF = orbital energies

#eps_ij = eps[:ndoublocc] #energies of doubly occupied orbitals
#eps_ab = eps[ndoubleocc:] #energies of virtual orbitals

#Retrieve ERI tensor from HF g = np.array(mints.ao_eri())
#g = np.array(mints.ao_eri())

#Retrieve MO coefficients from HF - need occupied and virtual - Cocc already stored in HF
def tensor_transf(nel, C, g):
    Cocc = C[:, :nel]
    Cvirt = C[:, nel:]

    temp = np.einsum("pi,pqrs->iqrs", Cocc, g)
    temp = np.einsum("rj,iqrs->iqjs", Cvirt, temp) 
    temp = np.einsum("qa,iqjs->iajs", Cocc, temp)
    gMO = np.einsum("sb,iajs->iajb", Cvirt, temp)


