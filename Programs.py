import functools
import multiprocessing
import numba
import numpy as np
import pandas as pd

from RadicalPair import *


# A simple function to transform a dense matrix into a sparse one

def Densify(Sparse_Matrix):
    return Sparse_Matrix.todense()

def Transform_to_eigenbasis(Matrix_of_eigenvectors, Matrix_to_transform):
    A = Matrix_of_eigenvectors
    A_inv = Matrix_of_eigenvectors.conj().T    # NB; if matrix is Hermitian then its inverse is its conjugate transpose

    return A_inv @ Matrix_to_transform @ A

def Plot_Spectrum(eigenvalues1, eigenvalues2 = None):
    fig, ax = plt.subplots()
    if eigenvalues2 is None:
        x = np.linspace(1, 2, 100)
        y = np.linspace(eigenvalues1, eigenvalues1, 100)
        ax.plot(x, y, color='red')

    else:
        x1 = np.linspace(1, 2.5, 100)
        y1 = np.linspace(eigenvalues1, eigenvalues1, 100)
        ax.plot(x1, y1, color='red')

        x2 = np.linspace(2.5, 4, 100)
        y2 = np.linspace(eigenvalues2, eigenvalues2, 100)
        ax.plot(x2, y2, color='black')


def Histogram_Height(Centre_Frequency, eigenvalues_ascending_order_MHz, Populations, Transition_Hamiltonian, Frequency_Bin_Width = 0.5):
    Histogram_Height = 0

    number_of_states = len(eigenvalues_ascending_order_MHz)

    for i in range(number_of_states):
        for j in range(number_of_states):
                if Centre_Frequency - 0.5*Frequency_Bin_Width <= np.abs(eigenvalues_ascending_order_MHz[i] - eigenvalues_ascending_order_MHz[j]) <= Centre_Frequency + 0.5*Frequency_Bin_Width:
                    Histogram_Height += np.square(Populations[i] - Populations[j])*np.square(np.abs(Transition_Hamiltonian[i, j]))

# NB !!! i've changed np.abs(Populations...) to np.square to match the change in the action-spectrum histograms that was mentioned in Jiate's paper !!
# Be careful to change this back if you want to get back to the calculations you have been doing before...

    return Histogram_Height

# Calculate an action spectrum by inputting the eigenvalues of the system (i.e. the involved energy levels), the populations of each state (which are
# the diagonal elements of the Ps matrix after transforming to the eigenbasis of H_0) and the transition probabilities (matrix elements of the Hamiltonian
# for the rf field after transforming to the eigenbasis of H_0)

# NB: the return of Action_Spectrum is an array Histogram_Heights which is the height of each histogram bar in the given frequency range

def Action_Spectrum(eigenvalues_ascending_order_MHz, Populations, Transition_Hamiltonian, Maximum_Frequency, display = False, Frequency_Bin_Width = 0.5, display_varying = False):
    Vmax = max(eigenvalues_ascending_order_MHz) - min(eigenvalues_ascending_order_MHz)
    Bins_Centres = np.linspace(0.5 * Frequency_Bin_Width, Maximum_Frequency+ 0.5 * Frequency_Bin_Width, int((Maximum_Frequency/Frequency_Bin_Width)+1))

    # Old code which calculates each bar height successively

    '''
    Histogram_Heights = []
    for v in np.linspace( 0.5 * Frequency_Bin_Width, Maximum_Frequency+ 0.5 * Frequency_Bin_Width, int((Maximum_Frequency/Frequency_Bin_Width)+1)):  # if Frequency_Bin_Width = 0.5 this is bins centred at [0.25, 0.75, 1.25, ..., 100.25]
        Histogram_Heights.append(Histogram_Height(v, eigenvalues_ascending_order_MHz, Populations, Transition_Hamiltonian, Frequency_Bin_Width=Frequency_Bin_Width))
    '''

    # We can use multiprocessing to parallelize the calculation of the various bar heights, using 4 (or more) CPU's

    pool = multiprocessing.Pool(4)
    Histogram_Heights = pool.map(functools.partial(Histogram_Height, eigenvalues_ascending_order_MHz=eigenvalues_ascending_order_MHz, Populations=Populations, Transition_Hamiltonian=Transition_Hamiltonian, Frequency_Bin_Width=Frequency_Bin_Width), Bins_Centres)


    return Histogram_Heights, Bins_Centres



# Histogram_Heights_Single_Field_Direction first generates H_0 (NB: for a static 50 uT field at a particular angle (theta,phi)) and then calculates the eigenvalues
# populations and transition matrix elements. It then uses Action_Spectrum as above.


def Histogram_Heights_Single_Field_Direction(theta,phi, Maximum_Frequency, rf_perpendicular = True, display = False, display_varying = False, dipolar = False, Frequency_Bin_Width = 0.5, rf_field_strength = 1e-5, Save_Histogram = False, File_Name = None):
    Hamiltonian = Densify(Sparse_Hamiltonian_total_basis(0.05, theta, phi, dipolar=dipolar))   # changed this so static field is always (0,0)...

    eigenvalues, eigenvectors = np.linalg.eigh(Hamiltonian)  # NB np.linalg.eigh gives [w[i]] eigenvalues in ascending order and a matrix with eigenvectors as COLUMNS not ROWS !!!
    eigenvalues = eigenvalues / 1e6   # Converting Eigenvalues to MHz

    Singlet_Projection_Matrix = eigenvectors.conj().T @ Singlet_Projection_total_basis() @ eigenvectors
    Populations = np.diag(Singlet_Projection_Matrix)

    if rf_perpendicular:
        Transition_Hamiltonian = eigenvectors.conj().T @ H_perp_total_basis(rf_field_strength, theta, phi) @ eigenvectors

    if not rf_perpendicular:
        Transition_Hamiltonian = eigenvectors.conj().T @ H_zee_total_basis(rf_field_strength, theta, phi) @ eigenvectors

    if display:
        Histogram_Heights, Bins_Centres  = Action_Spectrum(eigenvalues, Populations, Transition_Hamiltonian,Maximum_Frequency,Frequency_Bin_Width=Frequency_Bin_Width)[0], Action_Spectrum(eigenvalues, Populations, Transition_Hamiltonian,Maximum_Frequency,Frequency_Bin_Width=Frequency_Bin_Width)[1]
        plt.bar(Bins_Centres, Histogram_Heights, width=Frequency_Bin_Width, align='center', color='green')

    if display_varying:
        Histogram_Heights, Bins_Centres  = Action_Spectrum(eigenvalues, Populations, Transition_Hamiltonian,Maximum_Frequency,Frequency_Bin_Width=Frequency_Bin_Width)[0], Action_Spectrum(eigenvalues, Populations, Transition_Hamiltonian,Maximum_Frequency,Frequency_Bin_Width=Frequency_Bin_Width)[1]

        plt.cla()
        plt.bar(Bins_Centres, Histogram_Heights, width=Frequency_Bin_Width, align='center', color='green')
        plt.title(f'{theta}')
        plt.draw()
        time.sleep(0.01)
        plt.pause(0.001)

    else:
        Histogram_Heights, Bins_Centres  = Action_Spectrum(eigenvalues, Populations, Transition_Hamiltonian,Maximum_Frequency,Frequency_Bin_Width=Frequency_Bin_Width)[0], Action_Spectrum(eigenvalues, Populations, Transition_Hamiltonian,Maximum_Frequency,Frequency_Bin_Width=Frequency_Bin_Width)[1]

    if Save_Histogram:
        np.save(File_Name + '_HISTOGRAM_HEIGHTS', Histogram_Heights)
        np.save(File_Name + '_Bins_Centres', Bins_Centres)

    return Histogram_Heights, Bins_Centres



def ASH_Averaged_Over_Field_Directions(Maximum_Frequency, Number_of_Field_Directions, Frequency_Bin_Width=0.5, Save_Histogram=False, File_Name=None, dipolar=False, rf_perpendicular=True):
    Cumulative_Histogram_Heights = np.zeros((int((Maximum_Frequency / Frequency_Bin_Width) + 1),))
    for [theta, phi] in [list_of_field_directions[i] for i in range(Number_of_Field_Directions)]:
        np.add(Cumulative_Histogram_Heights, Histogram_Heights_Single_Field_Direction(theta, phi, Maximum_Frequency, dipolar = dipolar, Frequency_Bin_Width=Frequency_Bin_Width, rf_perpendicular=rf_perpendicular)[0], out=Cumulative_Histogram_Heights, casting="unsafe")
        # Cumulative_Histogram_Heights += Histogram_Heights_Single_Field_Direction(theta, phi, Maximum_Frequency, dipolar = dipolar)
        print(f'Completed: Theta = {theta} , Phi =  {phi}')

    Bins_Centres = np.linspace(0.5 * Frequency_Bin_Width, Maximum_Frequency + 0.5 * Frequency_Bin_Width,int((Maximum_Frequency / Frequency_Bin_Width) + 1))
    plt.bar(Bins_Centres, Cumulative_Histogram_Heights, width=Frequency_Bin_Width, align='center', color='green')

    if Save_Histogram:
        np.save(File_Name+'_HISTOGRAM_HEIGHTS', Cumulative_Histogram_Heights)
        np.save(File_Name+'_Bins_Centres', Bins_Centres)






















def Visualise_Matrix(Matrix, Matrix_Name = None):

    if scipy.sparse.issparse(Matrix):
        Matrix = Densify(Matrix)

    thing = np.array([Matrix[i, j] for i in range(int(np.sqrt(np.size(Matrix)))) for j in range(int(np.sqrt(np.size(Matrix))))])


    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121, projection='3d')

    _x = np.arange(np.sqrt(np.size(Matrix)))
    _y = np.arange(np.sqrt(np.size(Matrix)))

    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top = thing
    bottom = np.zeros_like(top)
    width = depth = 1

    ax1.bar3d(x, y, bottom, width, depth, top, shade=True)

    if Matrix_Name is not None:
        ax1.set_title(f'Matrix Elements of {str(Matrix_Name)}')


def List_Nuclei_Decreasing_Axiality():
        nucleus_axiality_list = [ (nucleus, ( (2*np.abs(nucleus.hyperfine_interaction_tensor[2,2])) / ( np.abs(nucleus.hyperfine_interaction_tensor[0,0]) + np.abs(nucleus.hyperfine_interaction_tensor[1,1])) )) for nucleus in RadicalA.all_A]
        nucleus_axiality_list = sorted(nucleus_axiality_list, key=lambda x:x[1])[::-1] # Sort the list of (nucleus, axiality) tuples in order of increasing axiality (NB: small axiality means highly polarised along z=axis in this case)
        nuclei_list = [nucleus_axiality_list[i][0] for i in range(len(nucleus_axiality_list))]

        return nuclei_list


def Vmax_comparison(List_of_Nuclei_in_subsystem_a, List_of_Nuclei_in_subsystem_b):
    for nucleus in List_of_Nuclei_in_subsystem_a:
        nucleus.add_to_simulation()

    Vmax_subsystem_a = Vmax(0.05, 0, 0)

    RadicalA.reset_simulation()
    RadicalA.remove_all_from_simulation()

    for nucleus in List_of_Nuclei_in_subsystem_b:
        nucleus.add_to_simulation()

    Vmax_subsystem_b = Vmax(0.00, 0, 0)

    RadicalA.reset_simulation()
    RadicalA.remove_all_from_simulation()

    return Vmax_subsystem_a + Vmax_subsystem_b

def Norm_of_Commutator():
    for nucleus in RadicalA.nuclei_included_in_simulation_A:
        if not nucleus.in_subsystem_b:
            nucleus.add_to_simulation()

    ha = Sparse_Hamiltonian_radicalA_basis(0.05,0,0)
    RadicalA.reset_simulation()
    RadicalA.remove_all_from_simulation()

    for nucleus in RadicalA.nuclei_included_in_simulation_A:
        if nucleus.in_subsystem_b:
            nucleus.add_to_simulation()

    hb = Sparse_Hamiltonian_radicalA_basis(0.00, 0, 0)
    RadicalA.reset_simulation()
    RadicalA.remove_all_from_simulation()

    Hb = scipy.sparse.kron(hb, RadicalA.Ia_dimension())

    Ha = scipy.sparse.kron(RadicalA.Ib_dimension(), ha)

    Norm_of_Commutator = scipy.sparse.linalg.norm(Ha * Hb - Hb * Ha, ord='fro') / np.sqrt(
        scipy.sparse.linalg.norm(Ha) * scipy.sparse.linalg.norm(Hb))


def Plot_Spectrum(eigenvalues1, eigenvalues2 = None):
    fig, ax = plt.subplots()
    if eigenvalues2 is None:
        x = np.linspace(1, 2, 100)
        y = np.linspace(eigenvalues1, eigenvalues1, 100)
        ax.plot(x, y, color='red')

    else:
        x1 = np.linspace(1, 2.5, 100)
        y1 = np.linspace(eigenvalues1, eigenvalues1, 100)
        ax.plot(x1, y1, color='red')

        x2 = np.linspace(2.5, 4, 100)
        y2 = np.linspace(eigenvalues2, eigenvalues2, 100)
        ax.plot(x2, y2, color='black')

def Number_of_Energy_Gaps(eigenvalues_ascending_order_MHz, Centre_Frequency):
    Number_of_Gaps = 0
    number_of_states = len(eigenvalues_ascending_order_MHz)

    for i in range(number_of_states):
        for j in range(number_of_states):
            if Centre_Frequency - 0.5 * 0.5 <= np.abs(
                    eigenvalues_ascending_order_MHz[i] - eigenvalues_ascending_order_MHz[
                        j]) <= Centre_Frequency + 0.5 * 0.5:
                Number_of_Gaps += 1

    return Number_of_Gaps


def Graph_of_Energy_Gaps(eigenvalues_ascending_order_MHz, Maximum_Frequency):
    fig2, ax2 = plt.subplots()

    Bar_Heights = []
    for v in np.linspace(0.25, Maximum_Frequency + 0.25, 2 * Maximum_Frequency + 1):
        Bar_Heights.append(Number_of_Energy_Gaps(eigenvalues_ascending_order_MHz,v))

    ax2.bar(np.linspace(0.25, Maximum_Frequency + 0.25, 2 * Maximum_Frequency + 1), Bar_Heights, width=0.5,
            align='center', color='green')


def Combine_subsystem_eigenvalues(eigenvalues_subsystem_a, eigenvalues_subsystem_b):
    sums_of_eigenvalues = []
    for eigenvalue2 in eigenvalues_subsystem_a:
        for eigenvalue1 in eigenvalues_subsystem_b:
            sums_of_eigenvalues.append(eigenvalue2 + eigenvalue1)

    return np.sort(sums_of_eigenvalues)


def Compare_Spectra_Exact_v_Subsystem():
    RadicalA.add_all_to_simulation()

    Hamiltonian = Densify(Sparse_Hamiltonian_total_basis(0.05, 0, 0))
    eigenvalues, eigenvectors = np.linalg.eigh(
        Hamiltonian)  # NB np.linalg.eigh() returns eigenvalues sorted in ASCENDING order, with the COLUMN eigenvectors[:,i] corresponding to eigenvalue w[i]
    eigenvectors = eigenvectors.T

    RadicalA.reset_simulation()
    RadicalA.remove_all_from_simulation()

    for nucleus in RadicalA.all_A:
        if nucleus.in_subsystem_b:
            nucleus.add_to_simulation()

    Hamiltonian_subsystem_b = Densify(Sparse_Hamiltonian_total_basis(0.05, 0, 0))
    eigenvalues_b, eigenvectors_b = np.linalg.eigh(Hamiltonian_subsystem_b)
    eigenvectors_b = eigenvectors_b.T

    RadicalA.reset_simulation()
    RadicalA.remove_all_from_simulation()

    for nucleus in RadicalA.all_A:
        if not nucleus.in_subsystem_b:
            nucleus.add_to_simulation()

    Hamiltonian_subsystem_a = Densify(Sparse_Hamiltonian_total_basis(0.0, 0, 0))
    eigenvalues_a, eigenvectors_a = np.linalg.eigh(Hamiltonian_subsystem_a)
    eigenvectors_a = eigenvectors_a.T

    eigenvalues_subsystem_method = Combine_subsystem_eigenvalues(eigenvalues_a, eigenvalues_b)

    Plot_Spectrum(eigenvalues, eigenvalues2=eigenvalues_subsystem_method)

    RadicalA.reset_simulation()
    RadicalA.remove_all_from_simulation()

    return eigenvalues, eigenvalues_subsystem_method


# A program to return automatically generated HFI tensorS (NB: in MHz !!!)

def tensor_list(number_of_tensors, axiality, off_diagonals=False, not_random=False, isotropic_random=False):
    list_of_tensors = []

    for _ in range(number_of_tensors):
        tensor = np.zeros((3, 3))
        tensor[0, 0] = random.uniform(-(1 - axiality), (1 - axiality)) * 10
        tensor[1, 1] = random.uniform(-(1 - axiality), (1 - axiality)) * 10
        tensor[2, 2] = random.uniform(-1, 1) * 10

        if off_diagonals:
            tensor[0, 1] = random.uniform(-(1 - axiality), (1 - axiality)) * 0.2
            tensor[1, 0] = random.uniform(-(1 - axiality), (1 - axiality)) * 0.2

            tensor[1, 2] = random.uniform(-(1 - axiality), (1 - axiality)) * 0.2
            tensor[2, 1] = random.uniform(-(1 - axiality), (1 - axiality)) * 0.2

            tensor[0, 2] = random.uniform(-(1 - axiality), (1 - axiality)) * 0.2
            tensor[2, 0] = random.uniform(-(1 - axiality), (1 - axiality)) * 0.2

        if not_random:
            tensor[0, 0] = 1 - axiality
            tensor[1, 1] = 1 - axiality
            tensor[2, 2] = axiality

        if isotropic_random:
            tensor[0, 0] = random.uniform(-1, 1) * 10
            tensor[1, 1] = random.uniform(-1, 1) * 10
            tensor[2, 2] = random.uniform(-1, 1) * 10

        tensor = np.multiply(tensor, 1e6)  # NB: Tensors are in MHz
        list_of_tensors.append(tensor)

    return list_of_tensors