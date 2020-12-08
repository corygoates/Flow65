"""Contains optimization methods for minimizing induced drag using sweep."""

import optix
import json

import numpy as np
import machupX as mx
import matplotlib.pyplot as plt
import scipy.optimize as opt


def drag_from_sweep_dist(sweeps, *args):
    """Calculates the induced drag on a swept wing at the given lift coef.

    Parameters
    ----------
    sweeps : ndarray
        Distribution of sweep angles along the wing. Assumed to be linearly spaced.

    taper : float
        Taper ratio.

    AR : float
        Aspect ratio.

    CL : float
        Desired lift coefficient.

    grid : int
        Nodes per semispan to use in MachUpX.

    Returns
    -------
    CD_i : float
        Induced drag coefficient.
    """

    # Get args
    taper = args[0]
    AR = args[1]
    CL = args[2]
    grid = args[3]

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
    
    # Set sweep
    print(sweeps)
    N_sweeps = len(sweeps)
    spans = np.linspace(0.0, 1.0, N_sweeps)
    sweep_list = [[float(span), float(sweep)*30.0] for span, sweep in zip(spans, sweeps)]
    wing_dict["wings"]["main_wing"]["sweep"] = sweep_list

    state = {
        "velocity" : 1.0
    }

    scene_input = {
        "solver" : {
            "max_iterations" : 1000
        }
    }

    # Initialize scene
    scene = mx.Scene(scene_input=scene_input)
    scene.add_aircraft("wing", wing_dict, state=state)

    # Get induced drag
    scene.target_CL(CL=CL, set_state=True)
    FM = scene.solve_forces(dimensional=False, body_frame=False)
    CD_i = FM["wing"]["total"]["CD"]
    print(CD_i)
    return CD_i


def model_optimum_wing(sweeps, *args):
    pass


if __name__=="__main__":

    # Optimize using gradient descent
    N_sweeps = 10
    s0 = np.zeros(N_sweeps)
    bounds = [(-1.0, 1.0)]*N_sweeps
    bounds = tuple(bounds)
    result = opt.minimize(drag_from_sweep_dist, s0, method='L-BFGS-B', args=(1.0, 8.0, 0.5, 40), bounds=bounds)
    print(result)