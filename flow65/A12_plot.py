"""Plots data for a NACA 2412 airfoil."""

import numpy as np
import matplotlib.pyplot as plt

from airfoil_tool import VortexPanelAirfoil

if __name__=="__main__":

    # Initialize alpha range and storage
    aL0 = -2.077
    a_v = np.linspace(-8.0, 20.0, 15)
    CL_v = np.zeros(15)

    # Declare experimental data
    a_e = [20.4370000, 18.2468000,16.3122000,15.7967000,15.2815000,14.2200000,12.0644000,10.0049000,7.9769700,5.8516700,3.7577400,1.6630400,-0.1747960,-2.3033500,-4.2074000,-6.4343300,-8.4368700]
    CL_e = [
        1.1337800,
        1.2089600,
        1.5095600,
        1.5650900,
        1.5944800,
        1.5275300,
        1.3984900,
        1.2432800,
        1.0438800,
        0.8558550,
        0.6481280,
        0.4566830,
        0.2455180,
        0.0473698,
        -0.1803510,
        -0.3753870,
        -0.6049370
    ]

    # Initialize airfoil
    airfoil = VortexPanelAirfoil(NACA="2412", x_le=0.0, x_te=1.0)
    airfoil.panel(99)

    # Get vortex panel results
    V = 1.0 # Arbitrary
    for i, a in enumerate(a_v):

        # Set condition
        airfoil.set_condition(alpha=a, V=V)

        # Get CL
        CL_v[i],_,_ = airfoil.solve()

    # Plot
    plt.figure()
    plt.plot(a_v, CL_v, '--r')
    plt.plot(a_v, 2.0*np.pi**2/180.0*(a_v-aL0), 'g-')
    plt.plot(a_e, CL_e, 'bo')
    plt.xlabel("Alpha [deg]")
    plt.ylabel("CL")
    plt.show()