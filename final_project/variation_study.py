"""Investigates the variation in induced drag with simple sweep parameters."""

import machupX as mx
import matplotlib.pyplot as plt
import numpy as np

def straight_swept_drag(sweep, taper, AR, alpha, grid):
    """Calculates the induced drag on a straight swept wing.

    Parameters
    ----------
    sweep : float
        Sweep angle in degrees.

    taper : float
        Taper ratio.

    AR : float
        Aspect ratio.

    alpha : float
        Angle of attack in degrees.

    grid : int
        Nodes per semispan to use in MachUpX.

    Returns
    -------
    CD_i : float
        Induced drag coefficient.
    """

    # Create MachUpX input
    wing_dict = {
        "weight" : 50.0,
        "units" : "English",
        "wings" : {
            "main_wing" : {
                "ID" : 1,
                "side" : "both",
                "is_main" : True,
                "semispan" : 1.5,
                "chord" : [[0.0, 1.0],
                           [1.0, taper]],
                "sweep" : sweep,
                "grid" : {
                    "N" : grid,
                }
            }
        }
    }

    state = {
        "velocity" : 1.0,
        "alpha" : alpha
    }

    # Initialize scene
    scene = mx.Scene()
    scene.add_aircraft("wing", wing_dict, state=state)

    # Get induced drag
    FM = scene.solve_forces(dimensional=False, body_frame=False)
    CD_i = FM["wing"]["total"]["CD"]
    return CD_i


if __name__=="__main__":

    # Straight swept wing
    N_sweeps = 21
    N_tapers = 6
    sweeps = np.linspace(-30.0, 30.0, N_sweeps)
    tapers = np.linspace(0.0, 1.0, N_tapers)
    CD_i = np.zeros((N_tapers, N_sweeps))
    plt.figure()
    for i, taper in enumerate(tapers):
        for j, sweep in enumerate(sweeps):

            CD_i[i,j] = straight_swept_drag(sweep, taper, 8.0, 5.0, 80)

        # Plot
        plt.plot(sweeps, CD_i[i,:], 'k-')
    plt.xlabel("Sweep Angle [deg]")
    plt.ylabel("Induced Drag Coefficient")
    plt.show()