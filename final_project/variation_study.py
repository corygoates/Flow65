"""Investigates the variation in induced drag with simple sweep parameters."""

import machupX as mx
import matplotlib.pyplot as plt
import numpy as np


def swept_drag(sweep, sweep_type, taper, AR, CL, grid, max_iter=100, verbose=False):
    """Calculates the induced drag on a swept wing at the given lift coef.

    Parameters
    ----------
    sweep : float or list
        Sweep angle in degrees.

    sweep_type : str
        May be "constant", "linear", or "jointed". If "jointed", then
        "sweep" should be a 2-element list.

    taper : float
        Taper ratio.

    AR : float
        Aspect ratio.

    CL : float
        Desired lift coefficient.

    grid : int
        Nodes per semispan to use in MachUpX.

    max_iter : int, optional
        Maximum iterations for the nonlinear solver in MachUpX. Defaults to 100.

    verbose : bool, optional
        Verbosity for MachUpX solver. Defaults to False.

    Returns
    -------
    CD_i : float
        Induced drag coefficient.
    """

    # Calculate semispan
    b_2 = 0.25*AR*(1.0+taper)

    # Create MachUpX input
    wing_dict = {
        "weight" : 50.0,
        "units" : "English",
        "wings" : {
            "main_wing" : {
                "ID" : 1,
                "side" : "both",
                "is_main" : True,
                "semispan" : b_2,
                "chord" : [[0.0, 1.0],
                           [1.0, taper]],
                "grid" : {
                    "N" : grid,
                }
            }
        }
    }
    if sweep_type=="constant":
        wing_dict["wings"]["main_wing"]["sweep"] = sweep
    elif sweep_type=="linear":
        wing_dict["wings"]["main_wing"]["sweep"] = [[0.0, 0.0],
                                                    [1.0, sweep]]
    elif sweep_type=="jointed":
        wing_dict["wings"]["main_wing"]["sweep"] = [[0.0, sweep[0]],
                                                    [0.5, sweep[0]],
                                                    [0.5, sweep[1]],
                                                    [1.0, sweep[1]]]

    state = {
        "velocity" : 1.0
    }

    # Initialize scene
    scene = mx.Scene(scene_input={"solver" : {"max_iterations" : max_iter}})
    scene.add_aircraft("wing", wing_dict, state=state)

    # Get induced drag
    scene.target_CL(CL=CL, set_state=True, verbose=verbose)
    FM = scene.solve_forces(dimensional=False, body_frame=False, verbose=verbose)
    CD_i = FM["wing"]["total"]["CD"]
    return CD_i


def sweep_and_taper_sweep(sweep_lims, N_sweeps, taper_lims, N_tapers, sweep_type, AR, CL_d, grid):
    """Performs a sweep in taper ratio and sweep angle and returns an array of the induced drag coefficients.

    Parameters
    ----------
    sweep_lims : list
        Limits of the sweep angle.

    N_sweeps : int
        Number of sweep angles to investigate.

    taper_lims : list
        Limits of the taper ratio.

    N_tapers : int
        Number of taper ratios to investigate.

    sweep_type : str
        Type of sweep to investigate

    AR : float
        Wing aspect ratio.

    CL_d : float
        Design lift coefficient.

    grid : int
        Nodes per semispan to use in MachUpX.

    Returns
    -------
    tapers : ndarray
        Array of taper ratios.

    sweeps : ndarray
        Array of sweep angles.

    CD_i : ndarray
        Array of induced drag coefficients.
    """

    # Initialize sweep variables
    sweeps = np.linspace(sweep_lims[0], sweep_lims[1], N_sweeps)
    tapers = np.linspace(taper_lims[0], taper_lims[1], N_tapers)

    # Initialize storage
    CD_i = np.zeros((N_tapers, N_sweeps))

    # Sweep
    for i, taper in enumerate(tapers):
        for j, sweep in enumerate(sweeps):

            # Get induced drag
            CD_i[i,j] = swept_drag(sweep, sweep_type, taper, AR, CL_d, grid)

    return tapers, sweeps, CD_i


if __name__=="__main__":

    # Investigate grid convergence for a straight swept wing

    # Initialize plotting
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8, 6))
    axs = axs.flatten()
    colors = ['#000000', '#222222', '#444444', '#666666', '#888888', '#AAAAAA', '#CCCCCC']
    grids = [10, 20, 40, 80]

    # Sweep
    for k, grid in enumerate(grids):
        
        tapers, sweeps, CD_i = sweep_and_taper_sweep([-30.0, 30.0], 21, [0.0, 1.0], 6, "constant", 8.0, 0.5, grid)

        # Plot
        for i, taper in enumerate(tapers):
            axs[k].plot(sweeps, CD_i[i,:], label=str(round(taper, 2)), color=colors[i])

        # Format subplot
        if k==1:
            axs[k].legend(title='Taper Ratio')
        if k>1:
            axs[k].set_xlabel("Sweep Angle [deg]")
        if k%2==0:
            axs[k].set_ylabel("CD_i")
        axs[k].set_title(str(grid)+" Nodes per Semispan")

    plt.savefig("final_project/plots/constant_sweep_and_taper_ratio_grid_study.png")


    # Straight swept wing
    
    # Get drag sweep
    tapers, sweeps, CD_i = sweep_and_taper_sweep([-30.0, 30.0], 21, [0.0, 1.0], 6, "constant", 8.0, 0.5, 80)

    # Plot
    plt.figure(figsize=(8, 6))
    for i, taper in enumerate(tapers):
        plt.plot(sweeps, CD_i[i,:], label=str(round(taper, 2)), color=colors[i])
    plt.xlabel("Sweep Angle [deg]")
    plt.ylabel("CD_i")
    plt.legend()
    plt.savefig("final_project/plots/constant_sweep_and_taper_ratio.png")


    # Crescent wing
    
    # Get drag sweep
    tapers, sweeps, CD_i = sweep_and_taper_sweep([-30.0, 30.0], 21, [0.0, 1.0], 6, "linear", 8.0, 0.5, 80)

    # Plot
    plt.figure(figsize=(8, 6))
    for i, taper in enumerate(tapers):
        plt.plot(sweeps, CD_i[i,:], label=str(round(taper, 2)), color=colors[i])
    plt.xlabel("Sweep Angle [deg]")
    plt.ylabel("CD_i")
    plt.legend()
    plt.savefig("final_project/plots/linear_sweep_and_taper_ratio.png")