"""Plots data for a NACA 2412 airfoil."""

import numpy as np
import matplotlib.pyplot as plt

import matplotlib

from airfoil_tool import VortexPanelAirfoil

if __name__=="__main__":

    # Initialize alpha range and storage
    aL0 = -2.077
    a_v = np.linspace(-16.0, 26.0, 22)
    CL_v = np.zeros(22)

    # Declare experimental data
    a_e = [
        24.515300,
        22.295800,
        20.306400,
        18.185900,
        16.288500,
        14.071300,
        12.265600,
        10.004300,
        8.194470,
        6.057190,
        4.082450,
        1.911920,
        -0.031767,
        -2.170220,
        -3.987940,
        -6.160960,
        -10.310200,
        -12.256900,
        -14.396600
    ]
    CL_e = [
        1.083100,
        1.061840,
        1.131360,
        1.169530,
        1.179620,
        1.229380,
        1.173390,
        1.039790,
        0.925950,
        0.726202,
        0.557802,
        0.353030,
        0.176306,
        -0.018620,
        -0.235073,
        -0.458164,
        -0.828423,
        -1.021860,
        -1.202110
    ]

    # Initialize airfoil
    airfoil = VortexPanelAirfoil(NACA="2421", x_le=0.0, x_te=1.0)
    airfoil.panel(99)

    # Get vortex panel results
    V = 1.0 # Arbitrary
    for i, a in enumerate(a_v):

        # Set condition
        airfoil.set_condition(alpha=a, V=V)

        # Get CL
        CL_v[i],_,_ = airfoil.solve()

    # Plot
    font = {'family' : 'serif',
            'size'   : 10}
    matplotlib.rc('font', **font)
    plt.figure(figsize=(4.25,4.25))
    plt.plot(a_v, CL_v, '--k', label="Vortex Panel")
    plt.plot(a_v, 2.0*np.pi**2/180.0*(a_v-aL0), 'k-', label="Thin-Airfoil")
    plt.plot(a_e, CL_e, 'ko', label="Experimental")
    plt.legend()
    plt.xlabel("Alpha [deg]")
    plt.ylabel("CL")
    plt.show()