"""Investigates the variation in induced drag with simple sweep parameters."""

import matplotlib

import machupX as mx
import matplotlib.pyplot as plt
import numpy as np


def swept_drag(**kwargs):
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

    max_iterations : int, optional
        Maximum iterations for the nonlinear solver in MachUpX. Defaults to 100.

    verbose : bool, optional
        Verbosity for MachUpX solver. Defaults to False.

    Returns
    -------
    CD_i : float
        Induced drag coefficient.
    """

    # Calculate semispan
    taper = kwargs["taper"]
    AR = kwargs["AR"]
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
                    "N" : kwargs["grid"],
                }
            }
        }
    }
    
    # Set sweep
    sweep = kwargs["sweep"]
    sweep_type = kwargs["sweep_type"]
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
    scene = mx.Scene(scene_input={"solver" : kwargs})
    scene.add_aircraft("wing", wing_dict, state=state)

    # Get induced drag
    verbose = kwargs.get("verbose", False)
    CL = kwargs["CL"]
    scene.target_CL(CL=CL, set_state=True, verbose=verbose)
    FM = scene.solve_forces(dimensional=False, body_frame=False, verbose=verbose)
    CD_i = FM["wing"]["total"]["CD"]
    return CD_i


def sweep_and_taper_sweep(sweep_lims, N_sweeps, taper_lims, N_tapers, **kwargs):
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

    CL : float
        Desired lift coefficient.

    grid : int
        Nodes per semispan to use in MachUpX.

    Returns
    -------
    tapers : ndarray
        Array of taper ratios.

    if sweep_type == "jointed":

        sweeps_0 : ndarray
            Array of inner sweep angles.

        sweeps_1 : ndarray
            Array of outer sweep angles.

    else:
        sweeps : ndarray
            Array of sweep angles.

    CD_i : ndarray
        Array of induced drag coefficients.
    """

    # Get taper ratios
    tapers = np.linspace(taper_lims[0], taper_lims[1], N_tapers)

    # Determine sweep type
    sweep_type = kwargs["sweep_type"]
    if sweep_type != "jointed":

        # Initialize sweep variables
        sweeps = np.linspace(sweep_lims[0], sweep_lims[1], N_sweeps)

        # Initialize storage
        CD_i = np.zeros((N_tapers, N_sweeps))

        # Sweep
        for i, taper in enumerate(tapers):
            for j, sweep in enumerate(sweeps):

                # Get induced drag
                CD_i[i,j] = swept_drag(sweep=sweep, taper=taper, **kwargs)

        return tapers, sweeps, CD_i

    else:

        # Initialize sweep variables
        sweeps_0 = np.linspace(sweep_lims[0][0], sweep_lims[0][1], N_sweeps)
        sweeps_1 = np.linspace(sweep_lims[1][0], sweep_lims[1][1], N_sweeps)

        # Initialize storage
        CD_i = np.zeros((N_tapers, N_sweeps, N_sweeps))

        # Sweep
        for i, taper in enumerate(tapers):
            for j, sweep_0 in enumerate(sweeps_0):
                for k, sweep_1 in enumerate(sweeps_1):

                    # Get induced drag
                    CD_i[i,j,k] = swept_drag(sweep=[sweep_0, sweep_1], taper=taper, **kwargs)

        return tapers, sweeps_0, sweeps_1, CD_i


if __name__=="__main__":

    # General vars
    colors = ['#000000', '#222222', '#444444', '#666666', '#888888', '#AAAAAA', '#CCCCCC']
    font = {
        "family" : "serif",
        "size" : 10
    }
    matplotlib.rc('font', **font)

    # Investigate grid convergence for a straight swept wing
    if False:

        # Initialize plotting
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8, 6))
        axs = axs.flatten()
        grids = [10, 20, 40, 80]

        # Sweep
        for k, grid in enumerate(grids):

            tapers, sweeps, CD_i = sweep_and_taper_sweep([-30.0, 30.0], 21, [0.0, 1.0], 6, sweep_type="constant", AR=8.0, CL=0.5, grid=grid)

            # Plot
            for i, taper in enumerate(tapers):
                axs[k].plot(sweeps, CD_i[i,:], label=str(round(taper, 2)), color=colors[i])

            # Format subplot
            if k==1:
                axs[k].legend(title='Taper Ratio')
            if k>1:
                axs[k].set_xlabel("Sweep Angle [deg]")
            if k%2==0:
                axs[k].set_ylabel("$CD_i$")
            axs[k].set_title(str(grid)+" Nodes per Semispan")

        plt.savefig("final_project/plots/constant_sweep_and_taper_ratio_grid_study.png")


    # Straight swept wing
    if False:
    
        # Get drag sweep
        tapers, sweeps, CD_i = sweep_and_taper_sweep([-30.0, 30.0], 21, [0.0, 1.0], 6, sweep_type="constant", AR=8.0, CL=0.5, grid=80)

        # Plot
        plt.figure(figsize=(6, 6))
        for i, taper in enumerate(tapers):
            plt.plot(sweeps, CD_i[i,:], label=str(round(taper, 2)), color=colors[i])
        plt.xlabel("Sweep Angle [deg]")
        plt.ylabel("$CD_i$")
        plt.legend(title="Taper Ratio")
        plt.savefig("final_project/plots/constant_sweep_and_taper_ratio_AR_8_CL_05.png")


    # Crescent wing
    if False:
    
        # Get drag sweep
        tapers, sweeps, CD_i = sweep_and_taper_sweep([-30.0, 30.0], 21, [0.0, 1.0], 6, sweep_type="linear", AR=8.0, CL=0.5, grid=80)

        # Plot
        plt.figure(figsize=(6, 6))
        for i, taper in enumerate(tapers):
            plt.plot(sweeps, CD_i[i,:], label=str(round(taper, 2)), color=colors[i])
        plt.xlabel("Sweep Angle [deg]")
        plt.ylabel("$CD_i$")
        plt.legend(title="Taper Ratio")
        plt.savefig("final_project/plots/linear_sweep_and_taper_ratio_AR_8_CL_05.png")

    
    # Jointed wing
    if True:
    
        # Get drag sweep
        tapers, sweeps_0, sweeps_1, CD_i = sweep_and_taper_sweep([[-30.0, 30.0],[-30.0, 30.0]], 11, [0.0, 1.0], 6, AR=8.0, CL=0.5, grid=40, sweep_type="jointed")

        # Plot
        for i, taper in enumerate(tapers):
            print(CD_i)
            plt.figure("$CD_i$ for $R_T$={0}".format(taper), figsize=(6, 6))
            plt.contour(sweeps_1, sweeps_0, CD_i[i])
            plt.xlabel("Outer Sweep Angle [deg]")
            plt.ylabel("Inner Sweep Angle [deg]")
            plt.show()