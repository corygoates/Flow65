"""Contains optimization methods for minimizing induced drag using sweep."""

import optix
import json
import matplotlib

import numpy as np
import machupX as mx
import matplotlib.pyplot as plt
import scipy.optimize as opt
import multiprocessing as mp


class DragCase:
    """
    Parameters
    ----------

    taper : float
        Taper ratio.

    AR : float
        Aspect ratio.

    CL : float
        Desired lift coefficient.

    grid : int
        Nodes per semispan to use in MachUpX.
    """

    def __init__(self, taper, AR, CL, grid):

        # Store params
        self._taper = taper
        self._AR = AR
        self._CL = CL
        self._N = grid


    def drag_from_sweep_dist(self, sweeps, *args, **kwargs):
        """Calculates the induced drag on a swept wing at the given lift coef.

        Parameters
        ----------
        sweeps : ndarray
            Distribution of sweep angles along the wing. Assumed to be linearly spaced.

        planform : bool, optional
            Whether to plot the planform. Defaults to False.

        Returns
        -------
        CD_i : float
            Induced drag coefficient.
        """

        # Calculate semispan
        b_2 = 0.25*self._AR*(1.0+self._taper)

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
                               [1.0, self._taper]],
                    "grid" : {
                        "N" : self._N,
                    }
                }
            }
        }

        # Set sweep
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
        if kwargs.get("planform", False):
            scene.display_planform(file_tag='_TR_{0}_AR_{1}_CL_{2}_N_sweep_{3}_N_grid_{4}'.format(self._taper, self._AR, self._CL, N_sweeps, self._N))

        # Get induced drag
        scene.target_CL(CL=self._CL, set_state=True)
        FM = scene.solve_forces(dimensional=False, body_frame=False)
        CD_i = FM["wing"]["total"]["CD"]
        return CD_i


def grad(x, *args):
    n = len(x)
    dx = 1e-6
    case = args[0]

    argslist = []
    for i in range(n):
        dx_v = np.zeros(n)
        dx_v[i] = dx
        argslist.append(x-dx_v)
        argslist.append(x+dx_v)

    with mp.Pool(100) as pool:
        results = pool.map(case.drag_from_sweep_dist, argslist)

    gradient = np.zeros(n)
    for i in range(n):
        gradient[i] = (results[2*i+1]-results[2*i])/(2*dx)
    return gradient


def optimize(s0, TR, AR, CL, N, method):
    """Optimizes a given wing in sweep distribution for induced drag.

    Parameters
    ----------
    s0 : ndarray
        Initial estiamte for sweep.

    TR : float
        Taper ratio

    AR : float
        Aspect ratio

    CL : float
        Lift coefficient

    N : int
        Nodes per semispan

    method : str
        Optimization method

    Returns
    -------
    CD_i

    K_D

    sweeps
    """

    # Get parameters
    N_sweeps = len(s0)
    bounds = [(-1.0, 1.0)]*N_sweeps
    bounds = tuple(bounds)

    # Display information
    print()
    print("Optimizing for:")
    print("    Taper ratio: {0}".format(TR))
    print("    Aspect ratio: {0}".format(AR))
    print("    Lift coef: {0}".format(CL))
    print("    Grid: {0}".format(N))
    print("    Sweep angles: {0}".format(N_sweeps))

    # Initialize case
    case = DragCase(TR, AR, CL, N)

    # Optimize
    result = opt.minimize(case.drag_from_sweep_dist, s0, method=method, jac=grad, args=(case,), bounds=bounds)
    print(result)
    CD_i = result.fun

    # Plot planform
    case.drag_from_sweep_dist(result.x, planform=True)

    # Plot sweep distribution
    spans = np.linspace(0.0, 1.0, N_sweeps)
    plt.figure(figsize=(5, 5))
    plt.plot(spans, result.x*30.0, 'k-')
    plt.xlabel("Span Fraction")
    plt.ylabel("Local Sweep Angle [deg]")
    plt.title("$CD_i$={0}".format(CD_i))
    plt.savefig('sweep_TR_{0}_AR_{1}_CL_{2}_N_sweep_{3}_N_grid_{4}.png'.format(TR, AR, CL, N_sweeps, N))
    
    # Calculate K_D
    K_D = (np.pi*case._AR*CD_i)/case._CL**2-1.0

    return CD_i, K_D, result.x*30.0


if __name__=="__main__":

    font = {
        "family" : "serif",
        "size" : 10
    }
    matplotlib.rc('font', **font)

    # Sweep space
    sweeps = [5, 10, 20]
    grids = [40, 80, 160]
    tapers = [1.0, 0.5, 0.0]
    aspects = [4.0, 8.0, 12.0, 16.0]
    CLs = [0.1, 0.3, 0.5]

    # Initialize file
    with open("optimization_results.txt", 'w') as file_handle:
        line = "{0:<15}{1:<10}{2:<15}{3:<15}{4:<10}{5:<30}{6:<30}".format("# Sweep Angles", "Grid", "Taper Ratio", "Aspect Ratio", "CL", "CD_i", "K_D")
        print(line, file=file_handle, flush=True)

        # Loop
        for N_sweeps in sweeps:
            for N in grids:
                for TR in tapers:
                    for AR in aspects:
                        for CL in CLs:

                            # Optimize
                            try:
                                CD_i, K_D = optimize(np.zeros(N_sweeps), TR, AR, CL, N, 'L-BFGS-B')

                                # Write results
                                line = "{0:<15}{1:<10}{2:<15}{3:<15}{4:<10}{5:<30}{6:<30}".format(N_sweeps, N, TR, AR, CL, CD_i, K_D)
                                print(line, file=file_handle, flush=True)
                            except:
                                continue