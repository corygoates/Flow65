"""Compares the analytic solutions for a Joukowski airfoil to estimations produced by vortex panel method."""

import os

import numpy as np
import matplotlib.pyplot as plt

from joukowski_airfoil_tool import JoukowskiAirfoil
from airfoil_tool import VortexPanelAirfoil


if __name__=="__main__":

    # Initialize Joukowski airfoil
    airfoil_j = JoukowskiAirfoil(design_CL=0.261, design_thickness=0.12, cylinder_radius=2.0)
    airfoil_j.set_condition(**{"freestream_velocity":10.0, "angle_of_attack[deg]":0.0})
    x_le = np.real(airfoil_j._z_le)
    x_te = np.real(airfoil_j._z_te)

    # Get analytic coefficients
    CL_a, Cm0_a, Cm_c4_a = airfoil_j.solve()

    # Initialize storage
    N_cases = 20
    CL = np.zeros(N_cases)
    Cm0 = np.zeros(N_cases)
    Cm_c4 = np.zeros(N_cases)

    # Loop through grid sizes
    grids = np.logspace(1, 3, num=N_cases).astype(int)
    for i, grid in enumerate(grids):

        # Export geometry
        filename = "airfoil_j_{0}.txt".format(grid)
        airfoil_j.export_geometry(grid, filename)

        # Load vortex panel airfoil
        airfoil_v = VortexPanelAirfoil(airfoil="file", filename=filename, x_te=x_te, x_le=x_le)
        airfoil_v.panel(grid-1)
        airfoil_v.set_condition(alpha=0.0, V=10.0)

        # Get coefficients
        CL[i], _, Cm_c4[i], Cm0[i] = airfoil_v.solve()

        # Cleanup
        os.remove(filename)

    # Plot CL
    plt.figure()
    plt.plot(grids, CL, 'ro')
    plt.plot(grids, np.ones_like(grids)*CL_a, 'b--')
    plt.xscale('log')
    plt.xlabel("Grid size")
    plt.ylabel("CL")
    plt.show()

    # Plot Cm0
    plt.figure()
    plt.plot(grids, Cm0, 'ro')
    plt.plot(grids, np.ones_like(grids)*Cm0_a, 'b--')
    plt.xscale('log')
    plt.xlabel("Grid size")
    plt.ylabel("Cm0")
    plt.show()

    # Plot Cm_c4
    plt.figure()
    plt.plot(grids, Cm_c4, 'ro')
    plt.plot(grids, np.ones_like(grids)*Cm_c4_a, 'b--')
    plt.xscale('log')
    plt.xlabel("Grid size")
    plt.ylabel("Cm_c4")
    plt.show()

    # Plot difference in CL
    CL_err = 100.0*np.abs(CL-CL_a)/CL_a
    plt.figure()
    plt.loglog(grids, CL_err)
    plt.xlabel("Grid size")
    plt.ylabel("% Error in CL")
    plt.show()