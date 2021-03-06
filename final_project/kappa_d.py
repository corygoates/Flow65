"""Investigates whether K_D is not a function of C_L within the G-H method."""

import matplotlib

import machupX as mx
import matplotlib.pyplot as plt
import numpy as np

from variation_study import swept_drag

def get_K_D(**kwargs):
    """Calculates the induced drag factor on a swept wing at the given lift coef.

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
    K_D : float
        Induced drag factor.
    """

    # Get induced drag
    CD_i = swept_drag(**kwargs)

    # Get relevant kwargs
    CL = kwargs["CL"]
    AR = kwargs["AR"]

    # Calculate K_D
    K_D = (np.pi*AR*CD_i)/CL**2-1.0
    return K_D


if __name__=="__main__":

    # Set figure format
    font = {
        "family" : "serif",
        "size" : 10
    }
    matplotlib.rc('font', **font)

    # Reproduce Fig. 1.8.14 from Phillips
    if False:

        # Declare sweep params
        ARs = np.linspace(4.0, 20.0, 9)
        N_tapers = 20
        tapers = np.linspace(0.0, 1.0, N_tapers)

        # Loop through aspect ratios
        plt.figure()
        for AR in ARs:
            print(AR)

            K_D = np.zeros(N_tapers)

            # Loop through taper ratios
            for i, taper in enumerate(tapers):

                # Get K_D
                K_D[i] = get_K_D(sweep=0.0, sweep_type="constant", taper=taper, AR=AR, CL=0.5, grid=40)

            plt.plot(tapers, K_D, label=str(int(AR)))

        plt.legend(title="Aspect Ratio")
        plt.xlabel("Taper Ratio")
        plt.ylabel("$\kappa_D$")
        plt.show()

    # Dependence of K_D on lift coef
    if False:

        N_sweeps = 3
        N_tapers = 3
        N_CLs = 11
        sweeps = np.linspace(-30.0, 30.0, N_sweeps)
        tapers = np.linspace(0.0, 1.0, N_tapers)
        CLs = np.linspace(0.01, 0.5, N_CLs)

        # Loop through tapers and sweeps
        for taper in tapers:
            for sweep in sweeps:

                # Loop through CLs
                K_D = np.zeros(N_CLs)
                for i, CL in enumerate(CLs):

                    K_D[i] = get_K_D(sweep=sweep, sweep_type="constant", taper=taper, AR=8.0, CL=CL, grid=40, verbose=False)

                plt.figure("{0} taper, {1} sweep".format(taper, sweep))
                plt.plot(CLs, K_D)
                plt.xlabel("$C_L$")
                plt.ylabel("$\kappa_D$")
                plt.show()

                # Determine max variation
                K_D_avg = np.average(K_D)
                variation = np.abs((K_D-K_D_avg)/K_D_avg)
                print("Max variation: {0}%".format(np.max(variation)*100))

    # Determine uswept K_D for sweep cases
    if True:

        # Sweep space
        tapers = [0.0, 0.5, 1.0]
        aspects = [4.0, 8.0, 12.0, 16.0]
        line = "{0:<15}{1:<15}{2:<30}".format("Aspect Ratio", "Taper Ratio", "K_D")
        print(line)

        for AR in aspects:
            for TR in tapers:
                K_D = get_K_D(sweep=0.0, sweep_type="constant", taper=TR, AR=AR, CL=0.5, grid=160, verbose=False)

                # Write results
                line = "{0:<15}{1:<15}{2:<30}".format(AR, TR, K_D)
                print(line)