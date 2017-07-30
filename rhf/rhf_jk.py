"""

The rhf class takes in molecule coordinates and basis_set as options
and calculates the hf energy
molecule = a string of ZMAT
basis_set = a string for the basis set

"""
import numpy as np
import psi4
#import diis_helper

class RHF:
    def __init__(self, mol, basis_set, options):

        #build a molecule
        self.mol = mol
        self.mol.update_geometry()

        #build a basis
        bas = psi4.core.BasisSet.build(self.mol, target=basis_set)
        #build the complementary JKFIT basis 
        aux = psi4.core.BasisSet.build(self.mol, fitrole = "JKFIT", other = basis_set)
        #build the zero basis set
        zero_bas = psi4.core.BasisSet.zero_ao_basis_set()
        #build a MintsHelper
        mints = psi4.core.MintsHelper(bas)

        self.nbf = mints.nbf()

        self.nelec = int(options['nelec']/2)

        #memory limitations
        if (self.nbf > 100):
            raise Exception("More than 100 basis functions!")

        #build (Q|ls) raw 3-index ERIs, dimensions (1, naux, nbf, nbf)
        Qls_tilde = mints.ao_eri(zero_bas, aux, bas, bas)
        Qls_tilde = np.squeeze(Qls_tilde) # remove the 1 - dimensions

        #build and invert coulomb metrix, dimension(1, aux, 1, aux)
        metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
        metric.power(-0.5, 1.e-14)
        metric = np.squeeze(metric) # remove the 1-dimensions

        #build the (P|ls) 
        self.Pls = np.einsum('Qls, PQ -> Pls', Qls_tilde, metric)

        V = np.array(mints.ao_potential())
        T = np.array(mints.ao_kinetic())

        self.g = np.array(mints.ao_eri())
        self.H = T + V
        self.S = np.array(mints.ao_overlap())
        self.A = mints.ao_overlap()
        self.A.power(-0.5, 1.e-14)
        self.A = np.array(self.A)

        self.E = 0.0
        self.e = np.zeros(len(self.H))
        self.C = np.zeros_like(self.H)

        #reading in the input parameters 
        self.e_conv = options['energy_conv']
        self.d_conv = options['density_conv']
        self.max_iter = options['max_iter']

        self.diis = options['diis']
        if self.diis == 'on':
            self.diis_start = options['diis_start']

        self.damping = options['damping']
        if self.damping == 'on':
            self.damp_value = options['damping_value']
            self.damp_start = options['damping_start']

        if (self.damping == 'on' and self.diis == 'on'):
            self.diis = 'off'
            raise Exception("Warning: cannot implement DIIS and damping simultaneously. Setting damping to false.")

    def damp(self,F_old, F):
        """
        This is the damping function that will return a Fock matrix
        Takes as input an old Fock matrix and the current one
        Returns  a linear combination using damping_value
        """

        F_new = self.damp_value * F_old + (1 - self.damp_value)*F
        return F_new

    def build_density(self, F):
        """
        This function builds the density matrix D
        Takes the fock matrix F as input, transforms, and diagonalizes
        Returns D, eigenvalues, and eigenvectors of F
        """

        Fp = (self.A).T@ F @ self.A
        eps, Cp  = np.linalg.eigh(Fp)
        C = self.A @ Cp
        Cocc = C[:,:self.nelec]
        D = Cocc @ Cocc.T
        return D, eps, C
        
    def build_JK(self, D):
        """
        This function builds the coulomb and exchange matrices using density fitting
        takes density as a parameter
        scales N^2 for J and and N^3 for  K
        """
        #build the coulomb matrix J
        X_P = np.einsum('Pls, ls -> P', self.Pls, D)
        J = np.einsum('Pmn,P-> mn', self.Pls, X_P)
    
        #build the exchange matrix K
        Z_Pnl = np.einsum('Pns, sl -> Pnl', self.Pls, D)
        K = np.einsum('Pml, Pnl -> mn', self.Pls, Z_Pnl)

        return J, K

    def build_JK_fast(self, Cocc, D):
        """
        This is an alternative function that builds the coulomb and exchange matrices using density fitting
        takes Cocc and D as a parameters
        scales N^2 for both J and K
        """
        #build the coulomb matrix J
        X_P = np.einsum('Pls, ls -> P', self.Pls, D)
        J = np.einsum('Pmn, P -> mn', self.Pls, X_P)
    
        #build the exchange matrix K
        Z_Pmp = np.einsum('Pms, sp -> Pmp', self.Pls, Cocc)
        Z_Pnp = np.einsum('Pnl, lp -> Pnp', self.Pls, Cocc)
        K = np.einsum('Pmp, Pnp -> mn', Z_Pmp, Z_Pnp)

        return J, K

    def get_energy(self):
        """
        This function calculates the RHF energy
        """

        D, self.e, self.C = self.build_density(self.H)

        E_old = 0.0
        F_old = None

        for iteration in range(self.max_iter):

            #J, K = self.build_JK_fast(self.C[:,:self.nelec],D)
            J, K = self.build_JK(D)

            # F = H_pq + 2 * G_pqrs D_rs - G_prqs D_rs
            F = self.H + 2.0 * J - K

            if self.damping == 'on' and iteration >= self.damp_start:
                F = self.damp(F_old, F)

            F_old = F

            if self.diis == 'on' and iteration >= self.diis_start:
                F = diis_helper.diis(F, D, self.diis_start,self.diis_vector)

            E_electric = np.sum((F + self.H) * D)
            self.E = E_electric + self.mol.nuclear_repulsion_energy()

            E_diff = self.E - E_old
            E_old = self.E

            # Build the AO gradient
            grad = F @ D @ self.S - self.S @ D @ F
            grad_rms = np.mean(grad  ** 2) ** 0.5

            D, self.e, self.C = self.build_density(F)

            print("Iteration: %3d Energy: % 16.12f  Energy difference: % 8.4f Gradient difference: % 8.4f" % (iteration, self.E, E_diff,grad_rms))

            if E_diff < self.e_conv and grad_rms < self.d_conv:
                break

        print("SCF has finished! \n")

if __name__ == '__main__':
    mol = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
    symmetry c1
    """
    )
    bas = 'cc-pvtz'

    options = {'energy_conv' : 1.0e-6, 'density_conv' : 1.0e-6,'max_iter': 25,
                'diis' : 'off', 'nelec' : 10, 'damping': 'on', 'damping_start' : 5, 'damping_value' : 0.2}

    molecule = RHF(mol, bas, options)
    molecule.get_energy()
    print(molecule.E)
    #psi4.set_options({"scf_type": "pk"})
    psi4_energy = psi4.energy("SCF/cc-pvtz", molecule=mol)

    print("Energy matches Psi4 %s" % np.allclose(psi4_energy, molecule.E))
