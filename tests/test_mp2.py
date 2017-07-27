"""
Testing for MP2.py
"""

import psi4
import pytest

def test_mp2():

	mol = psi4.geometry("""
	O
	H 1 1.1
	H 1 1.1 2 104
	symmetry c1
	"""
	)

	bas = 'sto-3g'

	scf_e, scf_wfn = psi4.energy('scf', return_wfn=TRUE)
	nel = scf_wfn.nalpha()

	eps = np.asarray(scf_wfn.epsilon_a())

	mints = psi4.core.MintsHelper(scf_wfnbasisset())
	g = np.asarray(mints.ao_eri())

	C = np.asarray(scf_wfn.Ca())
	
	assert tensor_transf(nel, C, g) == 
