import scipy.sparse.linalg

from RadicalA import *
from RadicalB import *


# NB: this is the dipolar tensor appropriate for a [FAD-TrpCH] radical pair in the ClCry4a (pigeon wild-type cryptochrome
# x-ray structure as described in the 2022 RF disruption paper

dipolar_tensor = np.array([[1.09,-12.12,4.91], [-12.12,-7.80,7.02], [4.91,7.02,6.71]])*1e6

# NB; The order of spins is {Nb1 Nb2 Nb3... EA Na1 Na2 Na3... } { Nb1 Nb2 Nb3... EA Na1 Na2 Na3...} - i.e. {RadicalA}{RadicalB}
# This allows for simplification of our spin systems to consider only a single radical (i.e. a FAD-Z type system)

# Further, we can subdivide a single radical into two subsystem {a} and {b} and calculate eigenvalues of a single
# subsystem (involving only {Nb1 Nb2 Nb3... EA} or only {EA Na1 Na2 Na3...} and if the Ha and Hb ~ commute (which occurs
# when the HFIT's involved are naturally polarised strongly along some axis, then the eigenvalues of the total system
# can be approximated as a sum of eigenvalues of the two subsystems - this follows from the Spectral Theorem, and in
# particular Weyl's inequality tells us that this process always overestimates Vmax


# Here we convert from the single-radical eigenbasis used in the classes {RadicalA, RadicalB} to construct
# the relevant matrices in the total basis including both radicals (though if we only include nuclei in one radical,
# this essentially just simplifies to adding a direct product of 𝟙2 to account for the second (bare) electron !!!

# For a nucleus in Radical B: I_vec = 𝟙A ⊗ i_vec
def I_vec_total_basis_B(B_nucleus):
    return np.array([scipy.sparse.kron(scipy.sparse.identity(RadicalA.I_radical_dimension(), format='coo'), B_nucleus.I_vec_single_radical_basis()[x]) for x in [0, 1, 2]])

# For a nucleus in Radical A: I_vec = i_vec ⊗ 𝟙B
def I_vec_total_basis_A(A_nucleus):
    return np.array([scipy.sparse.kron(A_nucleus.I_vec_single_radical_basis()[x], scipy.sparse.identity(RadicalB.I_radical_dimension(), format='coo')) for x in [0, 1, 2]])

# For electron B: S_vec = 𝟙A ⊗ s_vec
def S_vec_total_basis_B(RadicalB):
    return np.array([scipy.sparse.kron(scipy.sparse.identity(RadicalA.I_radical_dimension(), format='coo'), RadicalB.S_vec_single_radical_basis()[x]) for x in [0, 1, 2]])

# For electron A: S_vec = s_vec ⊗ 𝟙B
def S_vec_total_basis_A(RadicalA):
    return np.array([scipy.sparse.kron(RadicalA.S_vec_single_radical_basis()[x], scipy.sparse.identity(RadicalB.I_radical_dimension(), format='coo')) for x in [0, 1, 2]])

# We calculate H_zee as -γB·S_vec where here S_vec = S_vec_A + S_vec_B

def H_zee_total_basis(field_strength, theta, phi):
    S_vec = S_vec_total_basis_A(RadicalA) + S_vec_total_basis_B(RadicalB)
    r_vec = np.array([np.sin(theta)*np.cos(phi) , np.sin(theta)*np.sin(phi), np.cos(theta)])
    return (gyromag / (2 * np.pi)) * field_strength * np.sum([r_vec[i] * S_vec[i] for i in range(3)], axis=0)


def H_perp_total_basis(rf_field_strength, theta, phi):
    S_vec = S_vec_total_basis_A(RadicalA) + S_vec_total_basis_B(RadicalB)
    perp_vec = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), - np.sin(theta)]) # A normalised vector perpendicular to r_vec above...
    return (gyromag / (2 * np.pi)) * rf_field_strength * np.sum([perp_vec[i] * S_vec[i] for i in range(3)], axis=0)


# Each term in the H_hfi Hamiltonian is of the form S_vec_A/B · Ai · I_vec_i

def H_hfi_total_basis():
    H = 0
    S_vec_A = S_vec_total_basis_A(RadicalA)
    S_vec_B = S_vec_total_basis_B(RadicalB)

    for nucleus in RadicalA.nuclei_included_in_simulation_A:
        I_vec = I_vec_total_basis_A(nucleus)
        H += sum(nucleus.hyperfine_interaction_tensor[i, j] * S_vec_A[i] @ I_vec[j] for i in range(3) for j in range(3))

    for nucleus in RadicalB.nuclei_included_in_simulation_B:
        I_vec = I_vec_total_basis_B(nucleus)
        H += sum(nucleus.hyperfine_interaction_tensor[i, j] * S_vec_B[i] @ I_vec[j] for i in range(3) for j in range(3))

    return H

def H_dip_total_basis():
    S_vec_A = S_vec_total_basis_A(RadicalA)
    S_vec_B = S_vec_total_basis_B(RadicalB)

    H_dip = sum(dipolar_tensor[i,j] * S_vec_A[i] @ S_vec_B[j] for i in range(3) for j in range(3))

    return H_dip

# Can then calculate the full hamiltonian in the total basis

def Sparse_Hamiltonian_total_basis(field_strength, theta, phi, dipolar = False):
    H_tot = H_zee_total_basis(field_strength,theta,phi) + H_hfi_total_basis()
    if dipolar:
        print('adding dipolar!')
        H_tot += H_dip_total_basis()

    return H_tot

# Vmax then calculated as ever...

def Vmax(field_strength, theta, phi, display = False, display_eigenvalues = False, dipolar = False):
    if display:
        print(f' \n Field strength = {field_strength} mT , theta = {theta}, phi = {phi} \n __________________________________________________________________')

    Hspar = Sparse_Hamiltonian_total_basis(field_strength, theta, phi, dipolar=dipolar)

    if display:
        print(f'Sparse Hamiltonian created in {time.perf_counter() - startTime}s')
        print(f'Radical A: {RadicalA.nuclei_included_in_simulation_A}\n',f'Radical B: {RadicalB.nuclei_included_in_simulation_B}')

    valmax = scipy.sparse.linalg.eigsh(Hspar, k=1, M=None, sigma=None, which='LA', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=False, Minv=None, OPinv=None)
    valmin = scipy.sparse.linalg.eigsh(Hspar, k=1, M=None, sigma=None, which='SA', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=False, Minv=None, OPinv=None)
    Vmax = valmax - valmin
    Vmax = Vmax[0]/1e6    # Converting Vmax from Hz to MHz

    if display_eigenvalues:
      print(f'Maximum Eigenvalue = {valmax * 2 *np.pi}, Minimum Eigenvalue = {valmin * 2 * np.pi}') # Showing the eigenvalues in rad s^-1

    if display:
        print(f'Vmax with {len(RadicalA.nuclei_included_in_simulation_A)+len(RadicalB.nuclei_included_in_simulation_B)} nuclei = {Vmax} MHz')
        print(f'Time Taken = {time.perf_counter()-startTime}')
    return Vmax


# Singlet Projection is 1/4 𝟙 - S_vec_A · S_vec_B

def Singlet_Projection_total_basis():
    S_vec_A = S_vec_total_basis_A(RadicalA)
    S_vec_B = S_vec_total_basis_B(RadicalB)

    return 0.25*scipy.sparse.identity(RadicalA.I_radical_dimension() * RadicalB.I_radical_dimension()) - sum(S_vec_A[i] @ S_vec_B[i] for i in range(3))


#-----------------------------------------------------------------------------------------------------------------------#


def H_zee_radicalA_basis(field_strength, theta, phi):
    S_vec = RadicalA.S_vec_single_radical_basis()
    r_vec = np.array([np.sin(theta)*np.cos(phi) , np.sin(theta)*np.sin(phi), np.cos(theta)])
    return (-gyromag / (2 * np.pi)) * field_strength * np.dot(r_vec, S_vec)


def H_perp_radicalA_basis(rf_field_strength, theta, phi):
    S_vec = RadicalA.S_vec_single_radical_basis()
    perp_vec = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), - np.sin(theta)]) # A normalised vector perpendicular to r_vec above...
    return (-gyromag / (2 * np.pi)) * rf_field_strength * np.dot(perp_vec, S_vec)


# Each term in the H_hfi Hamiltonian is of the form S_vec_A/B · Ai · I_vec_i

def H_hfi_radicalA_basis():
    H = 0
    S_vec_A = RadicalA.S_vec_single_radical_basis()

    for A_nucleus in RadicalA.nuclei_included_in_simulation_A:
        I_vec = A_nucleus.I_vec_single_radical_basis()
        H += sum(A_nucleus.hyperfine_interaction_tensor[i, j] * S_vec_A[i] @ I_vec[j] for i in range(3) for j in range(3))

    return H

# Can then calculate the full hamiltonian in the total basis

def Sparse_Hamiltonian_radicalA_basis(field_strength, theta, phi, dipolar = False):
    H_tot = H_zee_radicalA_basis(field_strength,theta,phi) + H_hfi_radicalA_basis()
    if dipolar:
        H_tot += 0

    return H_tot

# Vmax then calculated as ever...

def Vmax_radicalA_basis(field_strength, theta, phi, display = False, display_eigenvalues = False, dipolar = False):
    if display:
        print(f' \n Field strength = {field_strength} mT , theta = {theta}, phi = {phi} \n __________________________________________________________________')

    Hspar = Sparse_Hamiltonian_radicalA_basis(field_strength, theta, phi, dipolar)

    if display:
        print(f'Sparse Hamiltonian created in {time.perf_counter() - startTime}s')
        print(f'Radical A: {RadicalA.nuclei_included_in_simulation_A}\n',f'Radical B: {RadicalB.nuclei_included_in_simulation_B}')

    valmax = scipy.sparse.linalg.eigsh(Hspar, k=1, M=None, sigma=None, which='LA', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=False, Minv=None, OPinv=None)
    valmin = scipy.sparse.linalg.eigsh(Hspar, k=1, M=None, sigma=None, which='SA', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=False, Minv=None, OPinv=None)
    Vmax = valmax - valmin
    Vmax = Vmax[0]/1e6    # Converting Vmax from Hz to MHz

    if display_eigenvalues:
      print(f'Maximum Eigenvalue = {valmax * 2 *np.pi}, Minimum Eigenvalue = {valmin * 2 * np.pi}') # Showing the eigenvalues in rad s^-1

    if display:
        print(f'Vmax with {len(RadicalA.nuclei_included_in_simulation_A)+len(RadicalB.nuclei_included_in_simulation_B)} nuclei = {Vmax} MHz')
        print(f'Time Taken = {time.perf_counter()-startTime}')
    return Vmax


# Singlet Projection is 1/4 𝟙 - S_vec_A · S_vec_B

def Singlet_Projection_total_basis():
    S_vec_A = S_vec_total_basis_A(RadicalA)
    S_vec_B = S_vec_total_basis_B(RadicalB)

    return 0.25*scipy.sparse.identity(RadicalA.I_radical_dimension() * RadicalB.I_radical_dimension()) - sum(S_vec_A[i] @ S_vec_B[i] for i in range(3))
