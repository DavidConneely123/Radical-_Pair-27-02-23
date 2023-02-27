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



# Defining our RadicalB class

class RadicalB:

    nuclei_included_in_simulation_B = []
    all_B = []
    isotopologues_B = []
    subsystem_b_B = []
    subsystem_a_B = []

    # Initialise an instance of the class (i.e. a single nucleus)

    def __init__(self, name: str, spin: float, hyperfine_interaction_tensor, is_isotopologue=False, in_subsystem_b = False):
        # Run validations to the received arguments
        assert spin % 0.5 == 0, f'Spin must be an integer or half-integer value !'

        # Bssigning properties to self object

        self.name = name
        self.spin = spin
        self.is_isotopologue = is_isotopologue
        self.hyperfine_interaction_tensor = hyperfine_interaction_tensor
        self.in_subsystem_b = in_subsystem_b

        # Actions to execute

        # Every time we initialise a new instance of the class it is added to the list 'all' and if it is a 13C it is
        # also added to the 'Isotopologues' list

        # NB! Initially no nuclei are included in the simulation

        RadicalB.all_B.append(self)

        # If the nucleus is an isotope or is in subsystem_b they are added to the relevant list

        if self.is_isotopologue:
            RadicalB.isotopologues_B.append(self)



    # This simply changes how the instance objects appear when we print RadicalB.all to the terminal


    def __repr__(self):
        return f"RadicalB('{self.name}', 'spin-{self.spin}') "

    ######################################################## Simulation Methods ##############################################

    def remove_from_simulation(self):
        try:
            RadicalB.nuclei_included_in_simulation_B.remove(self)

        except:
            raise Exception('The nucleus you tried to remove is not currently part of the simulation !')

    @classmethod
    def remove_all_from_simulation(cls):
        try:
            for nucleus in RadicalB.all_B:
                RadicalB.nuclei_included_in_simulation_B.remove(nucleus)
                RadicalB.subsystem_a_B = []
                RadicalB.subsystem_b_B = []

        except:
            raise Exception('The nucleus you tried to remove is not currently part of the simulation !')

    def add_to_simulation(self):
        RadicalB.nuclei_included_in_simulation_B.append(self)

        if self.in_subsystem_b:
            RadicalB.subsystem_b_B.append(self)

        if not self.in_subsystem_b:
            RadicalB.subsystem_a_B.append(self)

    @classmethod
    def add_all_to_simulation(cls):
        for nucleus in RadicalB.all_B:
            RadicalB.nuclei_included_in_simulation_B.append(nucleus)

            if nucleus.in_subsystem_b:
                RadicalB.subsystem_b_B.append(nucleus)

            if not nucleus.in_subsystem_b:
                RadicalB.subsystem_a_B.append(nucleus)

    @classmethod
    def reset_simulation(cls):
        RadicalB.nuclei_included_in_simulation_B = []
        for nucleus in RadicalB.all_B:
            RadicalB.nuclei_included_in_simulation_B.append(nucleus) # This adds all the nuclei to the simulation

    def deuterate(self):
        self.hyperfine_interaction_tensor = 0.154*self.hyperfine_interaction_tensor
        self.spin = 1
        self.name = self.name + '_DEUTERBTED'

    def nitrogen_label(self):
        self.hyperfine_interaction_tensor = -1.402*self.hyperfine_interaction_tensor
        self.spin = 1/2
        self.name = self.name + '_LBBELLED_15N'

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

    # We find the dimensions of the nuclear eigenbasis before and after the nucleus of interest

    @classmethod
    def Ib_dimension(cls):
        Ib_dimension = 1
        for nucleus in RadicalB.subsystem_b_B:
            if nucleus in RadicalB.nuclei_included_in_simulation_B:
                if nucleus.spin == 1 / 2:
                    Ib_dimension *= 2
                if nucleus.spin == 1:
                    Ib_dimension *= 3

        return Ib_dimension

    @classmethod
    def Ia_dimension(cls):
        Ia_dimension = 1
        for nucleus in RadicalB.subsystem_a_B:
            if nucleus in RadicalB.nuclei_included_in_simulation_B:
                if nucleus.spin == 1 / 2:
                    Ia_dimension *= 2
                if nucleus.spin == 1:
                    Ia_dimension *= 3

        return Ia_dimension

    @classmethod
    def I_radical_dimension(cls):
        return RadicalB.Ib_dimension()*2*RadicalB.Ia_dimension()

    def Ib_before_dimension(self):
        if self in RadicalB.subsystem_a_B:
            return RadicalB.Ib_dimension()

        else:
            Ib_before_length = 1
            for nucleus in RadicalB.nuclei_included_in_simulation_B[0: RadicalB.nuclei_included_in_simulation_B.index(self)]:
                if nucleus in RadicalB.subsystem_b_B:
                    if nucleus.spin == 1:
                        Ib_before_length *= 3
                    if nucleus.spin == 1 / 2:
                        Ib_before_length *= 2

            return Ib_before_length

    def Ib_after_dimension(self):
        if self in RadicalB.subsystem_a_B:
            return 1

        else:
            Ib_after_length = 1
            for nucleus in RadicalB.nuclei_included_in_simulation_B[RadicalB.nuclei_included_in_simulation_B.index(self) + 1:]:
                if nucleus in RadicalB.subsystem_b_B:
                    if nucleus.spin == 1:
                        Ib_after_length *= 3
                    if nucleus.spin == 1 / 2:
                        Ib_after_length *= 2

            return Ib_after_length

    def Ia_before_dimension(self):
        if self in RadicalB.subsystem_b_B:
            return 1

        else:
            Ia_before_length = 1
            for nucleus in RadicalB.nuclei_included_in_simulation_B[0: RadicalB.nuclei_included_in_simulation_B.index(self)]:
                if nucleus in RadicalB.subsystem_a_B:
                    if nucleus.spin == 1:
                        Ia_before_length *= 3
                    if nucleus.spin == 1 / 2:
                        Ia_before_length *= 2

            return Ia_before_length

    def Ia_after_dimension(self):
        if self in RadicalB.subsystem_b_B:
            return RadicalB.Ia_dimension()

        else:
            Ia_after_length = 1
            for nucleus in RadicalB.nuclei_included_in_simulation_B[RadicalB.nuclei_included_in_simulation_B.index(self) + 1:]:
                if nucleus in RadicalB.subsystem_a_B:
                    if nucleus.spin == 1:
                        Ia_after_length *= 3
                    if nucleus.spin == 1 / 2:
                        Ia_after_length *= 2

            return Ia_after_length

    # We can make matrix representations for the nuclear operators
    def I_vec_single_radical_basis(self):
        if self.in_subsystem_b:
            return np.array([scipy.sparse.kron(scipy.sparse.identity(self.Ib_before_dimension(), format='coo'), scipy.sparse.kron(spin_operator, scipy.sparse.kron(scipy.sparse.identity(self.Ib_after_dimension(), format='coo'), scipy.sparse.identity(self.Ia_dimension()*2, format='coo')))) for spin_operator in [self.sx(), self.sy(), self.sz()]])

        else:
            return np.array([scipy.sparse.kron(scipy.sparse.identity(self.Ib_dimension() * 2, format='coo'), scipy.sparse.kron(scipy.sparse.identity(self.Ia_before_dimension(), format='coo'), scipy.sparse.kron(spin_operator, scipy.sparse.identity(self.Ia_after_dimension(), format='coo')))) for spin_operator in [self.sx(), self.sy(), self.sz()]])

    @classmethod
    def S_vec_single_radical_basis(cls):
        return np.array([scipy.sparse.kron(scipy.sparse.identity(RadicalB.Ib_dimension(), format='coo'), scipy.sparse.kron(spin_operator, scipy.sparse.identity(RadicalB.Ia_dimension(), format ='coo'))) for spin_operator in [sx_spinhalf, sy_spinhalf, sz_spinhalf]])













