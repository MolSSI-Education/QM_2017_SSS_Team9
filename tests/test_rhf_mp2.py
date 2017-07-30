"""
Testing
"""
import rhf
import pytest 
import psi4
import numpy as np

def test_rhf_mp2():

    """
    This function tests the rhf module with MP2 correction
    """

    mol = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
    symmetry c1
    """
    )

    bas = 'cc-pvdz'

    options = {'energy_conv' : 1.0e-6, 'density_conv' : 1.0e-6, 'max_iter' : 25,
                'diis' : 'off', 'nelec' : 10, 'damping' : 'off', 'scs-mp2' : 'off'}

    molecule = rhf.RHF(mol, bas, options)
    molecule.get_energy()
    e_mp2 = rhf.mp2(molecule, molecule.E, options)
    psi4_energy = psi4.energy('mp2/'+bas, molecule = mol)
    assert np.allclose(psi4_energy, e_mp2, 1e-04)


def test_rhf_mp2_scs():

    """
    This function tests the rhf module with MP2 correction
    """

    mol = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
    symmetry c1
    """
    )

    bas = 'cc-pvdz'

    options = {'energy_conv' : 1.0e-6, 'density_conv' : 1.0e-6, 'max_iter' : 25,
                'diis' : 'off', 'nelec' : 10, 'damping' : 'off', 'scs-mp2' : 'on'}

    molecule = rhf.RHF(mol, bas, options)
    molecule.get_energy()
    e_mp2 = rhf.mp2(molecule, molecule.E, options)
    psi4_energy = psi4.energy('mp2/'+bas, molecule = mol)
    assert np.allclose(psi4_energy, e_mp2, 1e-04)

