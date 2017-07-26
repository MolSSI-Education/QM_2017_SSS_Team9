"""

The rhf class takes in molecule coordinates and basis_set as options 
and calculates the hf energy  
molecule = a string of ZMAT
basis_set = a string for the basis set 

""" 
import numpy as np
import psi4
#import diis_helper

class Rhf:
    def __init__(self, mol, basis_set, options):

        #build a moelcule 
        self.mol = mol
        self.mol.update_geometry()

        #build a basis
        self.bas = psi4.core.BasisSet.build(self.mol, target=basis_set)

        #build a MintsHelper
        mints = psi4.core.MintsHelper(self.bas)

        self.nbf = mints.nbf()

        self.nelec = -self.mol.molecular_charge()

        #memory limitations
        if (self.nbf > 100):
            raise Exception("More than 100 basis functions!")

        self.e_conv = options['energy_conv'] 
        self.d_conv = options['density_conv']
        self.max_iter = options['max_iter']
        self.diis = options['diis']
        if self.diis == 'on':
            self.diis_start = options['diis_start']
#        self.damping = options['damping']
#            if self.damping == 'on':
#                self.damp_value = ['damping_value']
#                self.damp_start = ['damping_start']
#
#        if (self.damping == 'on' and self.diis == 'on'):
#            self.diis = 'off'
#            raise Exception("Warning: cannot implement DIIS and damping simultaneously. Setting damping to false.")

        V = np.array(mints.ao_potential())
        T = np.array(mints.ao_kinetic())

        self.g = np.array(mints.ao_eri())
        self.H = T + V
        self.S = np.array(mints.ao_overlap())
        self.A = np.array(mints.ao_overlap().power(-0.5, 1.e-14))

        self.E = 0.0
        self.e = np.zeros(len(self.H))
        self.C = np.zeros_like(self.H)

    def build_density(self, F):

        #Fp = (self.A).T @ F @ self.A
        temp = self.A.T @ F
        Fp = temp @ self.A
        eps, Cp  = np.linalg.eigh(Fp)
        C = self.A @ Cp
        Cocc = C[:,:self.nelec]
        D = Cocc @ Cocc.T
        return D, eps, C

    def get_energy(self):

        D, self.e, self.C = self.build_density(self.H) 

        E_old = 0.0
        F_old = None

        for iteration in range(max_iter):

            J = np.einsum("pqrs, rs -> pq", self.g, D)
            K = np.einsum("prqs, rs -> pq", self.g, D)

            # F = H_pq + 2 * G_pqrs D_rs - G_prqs D_rs
            F = self.H + 2.0 * J - K

            if self.diis == 'on' and iteration >= self.diis_start:
                F = diis_helper.diis(F, D, self.diis_start,self.diis_vector)

            D, self.e, self.C = self.build_density(F) 

            E_electric = np.sum((F + self.H) * D)
            self.E = E_electric + mol.nuclear_repulsion_energy()

            E_diff = self.E - E_old
            E_old = self.E

            # Build the AO gradient
            grad = F @ D @ self.S - self.S @ D @ F
            grad_rms = np.mean(grad  ** 2) ** 0.5  

            print("Iteration: %3d Energy: % 16.12f  Energy difference: % 8.4f" % (iteration, self.E, E_diff))
            if E_diff < e_conv and grad_rms < d_conv:
                break

        print("SCF has finished! \n")

        #psi4.set_options({"scf_type": "pk"})
        #psi4_energy = psi4.energy("SCF/sto-3g", molecule=mol)
        #print("Energy matches Psi4 %s" % np.allclose(psi4_energy, E_total))

if __name__ == '__main__':
    mol = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
    symmetry c1
    """
    )

    bas = 'sto-3g'

    options = {'energy_conv' : 1.0e-6, 'density_conv' : 1.0e-6,'max_iter': 25,
                'diis' : 'off'} 

    molecule = Rhf(mol, bas, options)
    molecule.get_energy()                                           
