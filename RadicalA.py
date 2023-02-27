import itertools
import random
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import scipy
import scipy.sparse
import scipy.sparse.linalg
from scipy.constants import physical_constants
import scipy.stats
import pandas as pd
import operator
import plotly.graph_objects as go


startTime = time.perf_counter()

i2 = scipy.sparse.identity(2)
sx_spinhalf = np.array([[0 + 0j, 0.5 + 0j], [0.5 + 0j, 0 + 0j]])
sy_spinhalf = np.array([[0 + 0j, 0 - 0.5j], [0 + 0.5j, 0 + 0j]])
sz_spinhalf = np.array([[0.5 + 0j, 0 + 0j], [0 + 0j, -0.5 + 0j]])

i3 = scipy.sparse.identity(3)
sx_spin1 = 1 / 2 * np.sqrt(2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
sy_spin1 = 1 / 2 * np.sqrt(2) * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])
sz_spin1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])

gyromag = scipy.constants.physical_constants['electron gyromagn. ratio'][0] / 1000   # NB ! in rad s^-1 mT^-1
i4 = scipy.sparse.identity(4)


TrpH_HFITs = np.load('../TrpH Calculations/CSVs and NPYs/TrpH_HFITS_MHz_ROTATED.npy')

field_directions_df = pd.read_csv('Alice ASH data/field_directions.csv')

list_of_field_directions = [entry for entry in tuple(field_directions_df.values)]




# Defining our RadicalA class

class RadicalA:

    # We keep list of instances of nuclei in RadicalA and which subsystems they are in

    nuclei_included_in_simulation_A = []
    all_A = []
    isotopologues_A = []
    subsystem_b_A = []
    subsystem_a_A = []

    # Initialise an instance of the class (i.e. a single nucleus)

    def __init__(self, name: str, spin: float, hyperfine_interaction_tensor, is_isotopologue=False, in_subsystem_b = False):
        # Run validations to the received arguments
        assert spin % 0.5 == 0, f'Spin must be an integer or half-integer value !'

        # Assigning properties to self object

        self.name = name
        self.spin = spin
        self.is_isotopologue = is_isotopologue
        self.hyperfine_interaction_tensor = hyperfine_interaction_tensor
        self.in_subsystem_b = in_subsystem_b

        # Actions to execute

        # Every time we initialise a new instance of the class it is added to the list 'all_A' and if it is a 13C it is
        # also added to the 'Isotopologues' list

        # NB! Initially no nuclei are included in the simulation

        RadicalA.all_A.append(self)

        # If the nucleus is an isotope or is in subsystem_b they are added to the relevant list

        if self.is_isotopologue:
            RadicalA.isotopologues_A.append(self)



    # This simply changes how the instance objects appear when we print RadicalA.all to the terminal


    def __repr__(self):
        return f"RadicalA('{self.name}', 'spin-{self.spin}') "

    ######################################################## Simulation Methods ##############################################

    def remove_from_simulation(self):
        try:
            RadicalA.nuclei_included_in_simulation_A.remove(self)

        except:
            raise Exception('The nucleus you tried to remove is not currently part of the simulation !')

    @classmethod
    def remove_all_from_simulation(cls):
        try:
            for nucleus in RadicalA.all_A:
                RadicalA.nuclei_included_in_simulation_A.remove(nucleus)
                RadicalA.subsystem_a_A = []
                RadicalA.subsystem_b_A = []

        except:
            raise Exception('The nucleus you tried to remove is not currently part of the simulation !')

    def add_to_simulation(self):
        RadicalA.nuclei_included_in_simulation_A.append(self)

        if self.in_subsystem_b:
            RadicalA.subsystem_b_A.append(self)

        if not self.in_subsystem_b:
            RadicalA.subsystem_a_A.append(self)

    @classmethod
    def add_all_to_simulation(cls):
        for nucleus in RadicalA.all_A:
            RadicalA.nuclei_included_in_simulation_A.append(nucleus)

            if nucleus.in_subsystem_b:
                RadicalA.subsystem_b_A.append(nucleus)

            if not nucleus.in_subsystem_b:
                RadicalA.subsystem_a_A.append(nucleus)

    @classmethod
    def reset_simulation(cls):
        RadicalA.nuclei_included_in_simulation_A = []
        for nucleus in RadicalA.all_A:
            RadicalA.nuclei_included_in_simulation_A.append(nucleus) # This adds all the nuclei to the simulation

    def deuterate(self):
        self.hyperfine_interaction_tensor = 0.154*self.hyperfine_interaction_tensor
        self.spin = 1
        self.name = self.name + '_DEUTERATED'

    def nitrogen_label(self):
        self.hyperfine_interaction_tensor = -1.402*self.hyperfine_interaction_tensor
        self.spin = 1/2
        self.name = self.name + '_LABELLED_15N'

    def move_to_subsystem_b(self):
        self.in_subsystem_b = True


    def identity_matrix(self):
        if self.spin == 1 / 2:
            return i2
        if self.spin == 1:
            return i3

    def sx(self):
        if self.spin == 1 / 2:
            return sx_spinhalf
        if self.spin == 1:
            return sx_spin1

    def sy(self):
        if self.spin == 1 / 2:
            return sy_spinhalf
        if self.spin == 1:
            return sy_spin1

    def sz(self):
        if self.spin == 1 / 2:
            return sz_spinhalf
        if self.spin == 1:
            return sz_spin1

    # Dimension of nuclear basis {Nb1 Nb2 Nb3...}

    @classmethod
    def Ib_dimension(cls):
        Ib_dimension = 1
        for nucleus in RadicalA.subsystem_b_A:
            if nucleus in RadicalA.nuclei_included_in_simulation_A:
                if nucleus.spin == 1 / 2:
                    Ib_dimension *= 2
                if nucleus.spin == 1:
                    Ib_dimension *= 3

        return Ib_dimension

    # Dimension of nuclear basis {Na1 Nb2 Nb3...}

    @classmethod
    def Ia_dimension(cls):
        Ia_dimension = 1
        for nucleus in RadicalA.subsystem_a_A:
            if nucleus in RadicalA.nuclei_included_in_simulation_A:
                if nucleus.spin == 1 / 2:
                    Ia_dimension *= 2
                if nucleus.spin == 1:
                    Ia_dimension *= 3

        return Ia_dimension

    # Dimension for basis of single radical {Nb1 Nb2 Nb3... EA Na1 Na2 Na3...}

    @classmethod
    def I_radical_dimension(cls):
        return RadicalA.Ib_dimension()*2*RadicalA.Ia_dimension()

    # Find the dimensions of the nuclear subsystem bases before and after the given nucleus

    def Ib_before_dimension(self):
        if self in RadicalA.subsystem_a_A:
            return RadicalA.Ib_dimension()   # all the b nuclei appear before the a nuclei

        else:
            Ib_before_length = 1
            for nucleus in RadicalA.nuclei_included_in_simulation_A[0: RadicalA.nuclei_included_in_simulation_A.index(self)]:
                if nucleus in RadicalA.subsystem_b_A:
                    if nucleus.spin == 1:
                        Ib_before_length *= 3
                    if nucleus.spin == 1 / 2:
                        Ib_before_length *= 2

            return Ib_before_length

    def Ib_after_dimension(self):
        if self in RadicalA.subsystem_a_A:
            return 1        # all the b nuclei appear before the a nuclei

        else:
            Ib_after_length = 1
            for nucleus in RadicalA.nuclei_included_in_simulation_A[RadicalA.nuclei_included_in_simulation_A.index(self) + 1:]:
                if nucleus in RadicalA.subsystem_b_A:
                    if nucleus.spin == 1:
                        Ib_after_length *= 3
                    if nucleus.spin == 1 / 2:
                        Ib_after_length *= 2

            return Ib_after_length

    def Ia_before_dimension(self):
        if self in RadicalA.subsystem_b_A:
            return 1      # all the b nuclei appear before the a nuclei

        else:
            Ia_before_length = 1
            for nucleus in RadicalA.nuclei_included_in_simulation_A[0: RadicalA.nuclei_included_in_simulation_A.index(self)]:
                if nucleus in RadicalA.subsystem_a_A:
                    if nucleus.spin == 1:
                        Ia_before_length *= 3
                    if nucleus.spin == 1 / 2:
                        Ia_before_length *= 2

            return Ia_before_length

    def Ia_after_dimension(self):
        if self in RadicalA.subsystem_b_A:
            return RadicalA.Ia_dimension()  # all the b nuclei appear before the a nuclei

        else:
            Ia_after_length = 1
            for nucleus in RadicalA.nuclei_included_in_simulation_A[RadicalA.nuclei_included_in_simulation_A.index(self) + 1:]:
                if nucleus in RadicalA.subsystem_a_A:
                    if nucleus.spin == 1:
                        Ia_after_length *= 3
                    if nucleus.spin == 1 / 2:
                        Ia_after_length *= 2

            return Ia_after_length

    # We can make matrix representations for the nuclear operators (NB: in the eigenbasis for the whole radical, but if there are
    # only nuclei in one subsystem then this reduces to the matrix representations in the eigenbasis of that subsystem automatically !!

    def I_vec_single_radical_basis(self):
        if self.in_subsystem_b:
            # 𝟙b,before ⊗ spin_operator  ⊗ 𝟙b,after ⊗ 𝟙2 ⊗ 𝟙a
            return np.array([scipy.sparse.kron(scipy.sparse.identity(self.Ib_before_dimension(), format='coo'), scipy.sparse.kron(spin_operator, scipy.sparse.kron(scipy.sparse.identity(self.Ib_after_dimension(), format='coo'), scipy.sparse.identity(self.Ia_dimension()*2, format='coo')))) for spin_operator in [self.sx(), self.sy(), self.sz()]])

            # 𝟙b ⊗ 𝟙2 ⊗ 𝟙a,before ⊗ spin_operator ⊗ 𝟙a,after
        else:
            return np.array([scipy.sparse.kron(scipy.sparse.identity(self.Ib_dimension() * 2, format='coo'), scipy.sparse.kron(scipy.sparse.identity(self.Ia_before_dimension(), format='coo'), scipy.sparse.kron(spin_operator, scipy.sparse.identity(self.Ia_after_dimension(), format='coo')))) for spin_operator in [self.sx(), self.sy(), self.sz()]])

    @classmethod
    def S_vec_single_radical_basis(cls):
        # 𝟙b ⊗ spin_operator ⊗ 𝟙a,
        return np.array([scipy.sparse.kron(scipy.sparse.identity(RadicalA.Ib_dimension(), format='coo'), scipy.sparse.kron(spin_operator, scipy.sparse.identity(RadicalA.Ia_dimension(), format ='coo'))) for spin_operator in [sx_spinhalf, sy_spinhalf, sz_spinhalf]])












