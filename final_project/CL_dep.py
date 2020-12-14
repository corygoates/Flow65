"""Investigates how the optimal sweep distribution is dependent on lift coefficient."""

import optix
import json
import matplotlib

import numpy as np
import machupX as mx
import matplotlib.pyplot as plt
import scipy.optimize as opt
import multiprocessing as mp

from optimization import DragCase, grad, optimize


if __name__=="__main__":

    font = {
        "family" : "serif",
        "size" : 10
    }
    matplotlib.rc('font', **font)

    # Params
    N_CLs = 11
    TR = 1.0
    AR = 12.0
    N = 80
    N_sweeps = 20
    CLs = np.linspace(0.1, 0.5, N_CLs)
    sweep_dists = np.zeros((N_CLs, N_sweeps))

    color_range = np.linspace(0, 155, N_CLs)
    colors = ["#"+"".join([hex(int(x)).replace('0x', '')]*3) for x in color_range]

    for i, CL in enumerate(CLs):
        _,_,sweep_dists[i] = optimize(np.zeros(N_sweeps), TR, AR, CL, N, 'L-BFGS-B')

    # Plot sweep distribution
    spans = np.linspace(0.0, 1.0, N_sweeps)
    plt.figure(figsize=(5, 5))
    for i, CL in enumerate(CLs):
        plt.plot(spans, sweep_dists[i], label=str(round(CL, 3)), color=colors[i])
    plt.xlabel("Span Fraction")
    plt.ylabel("Local Sweep Angle [deg]")
    plt.legend(title="$C_L$")
    plt.show()