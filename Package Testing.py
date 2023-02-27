import numpy as np

from RadicalPair import *
from Programs import *

N5 = RadicalA('N5', 1, np.array([[-2.79, -0.08, 0], [-0.08, -2.45, 0], [0, 0, 49.24]]) * 1e6)
NE1 = RadicalB('NE1', 1, np.array([[-1.48,1.64,-1.29],[1.64,15.82,-15.83],[-1.29,-15.83,12.70]])*1e6)

RadicalA.add_all_to_simulation()
RadicalB.add_all_to_simulation()

fig, ax = plt.subplots()


ASH_Averaged_Over_Field_Directions(1, 100, 0.001)



plt.show()


