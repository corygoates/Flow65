"""Compares the analytic solutions for a Joukowski airfoil to estimations produced by vortex panel method."""

import os

import numpy as np
import matplotlib.pyplot as plt

from joukowski_airfoil_tool import JoukowskiAirfoil
from airfoil_tool import VortexPanelAirfoil


if __name__=="__main__":

    # Initialize analytic airfoil
    airfoil_a = VortexPanelAirfoil(airfoil="UL08", trailing_edge="closed", n_points=10, CL_design=0.25)

    # Initialize storage
    N_cases = 100
    grids = np.unique(np.logspace(1, 3, num=N_cases).astype(int))
    CL = np.zeros(grids.size)
    Cm0 = np.zeros(grids.size)
    Cm_c4 = np.zeros(grids.size)

    # Loop through grid sizes
    for i, grid in enumerate(grids):

        # Export geometry
        filename = airfoil_a._airfoil+".txt"
        airfoil_a.panel(grid)
        airfoil_a.export_geometry()

        # Load vortex panel airfoil
        airfoil_v = VortexPanelAirfoil(airfoil="file", filename=filename)
        airfoil_v.set_condition(alpha=0.0, V=10.0)

        # Get coefficients
        CL[i], _, Cm_c4[i], Cm0[i] = airfoil_v.solve()
        #airfoil_v.plot_streamlines(-5.0, [-5.0, 5.0], 0.05, 10, 0.25)

        # Cleanup
        os.remove(filename)

    # Plot CL
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, sharex=True)
    ax0.plot(grids, CL, 'ro')
    ax0.set_xscale('log')
    ax0.set_ylabel("CL")

    # Plot Cm_c4
    ax1.plot(grids, Cm_c4, 'ro')
    ax1.set_xscale('log')
    ax1.set_ylabel("Cm_c4")

    ## Plot difference in CL
    #CL_err = 100.0*np.abs((CL-CL_a)/CL_a)
    #ax2.loglog(grids, CL_err)
    #ax2.set_xlabel("Grid size")
    #ax2.set_ylabel("% Error in CL")

    ## Plot difference in Cm_c4
    #Cm_c4_err = 100.0*np.abs((Cm_c4-Cm_c4_a)/Cm_c4_a)
    #ax3.loglog(grids, Cm_c4_err)
    #ax3.set_xlabel("Grid size")
    #ax3.set_ylabel("% Error in Cm_c4")

    plt.show()