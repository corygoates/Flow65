import numpy as np
import matplotlib.pyplot as plt

from wing_tool import Wing

if __name__=="__main__":

    # PART 1
    # Initialize wing with no washout
    wing1 = Wing(planform="elliptic",
                 AR=8.0)
    wing1.set_grid(50)

    # Initialize wing with linear washout
    wing2 = Wing(planform="elliptic",
                 AR=8.0,
                 washout="linear",
                 washout_mag=5.0)
    wing2.set_grid(50)

    # Sweep
    N_alpha = 101
    alphas = np.linspace(-15.0, 15.0, N_alpha)
    CL1 = np.zeros(N_alpha)
    CD1 = np.zeros(N_alpha)
    CL2 = np.zeros(N_alpha)
    CD2 = np.zeros(N_alpha)
    for i, alpha in enumerate(alphas):

        # Set condition
        wing1.set_condition(alpha=alpha)
        wing2.set_condition(alpha=alpha)

        # Solve
        wing1.solve()
        wing2.solve()
        CL1[i] = wing1.CL
        CD1[i] = wing1.CD_i
        CL2[i] = wing2.CL
        CD2[i] = wing2.CD_i

    # Plot
    plt.figure()
    plt.plot(CL1, CD1, 'k--', label='No Washout')
    plt.plot(CL2, CD2, 'k-', label='Linear Washout')
    plt.xlabel("CL")
    plt.ylabel("CDi")
    plt.title("Elliptic Planform")
    plt.legend()
    plt.show()

    # PART 1
    # Initialize wing with no washout
    wing1 = Wing(planform="tapered",
                 AR=8.0,
                 RT=0.5)
    wing1.set_grid(50)

    # Initialize wing with linear washout
    wing2 = Wing(planform="tapered",
                 AR=8.0,
                 RT=0.5,
                 washout="linear",
                 washout_mag=5.0)
    wing2.set_grid(50)

    # Sweep
    N_alpha = 101
    alphas = np.linspace(-15.0, 15.0, N_alpha)
    CL1 = np.zeros(N_alpha)
    CD1 = np.zeros(N_alpha)
    CL2 = np.zeros(N_alpha)
    CD2 = np.zeros(N_alpha)
    for i, alpha in enumerate(alphas):

        # Set condition
        wing1.set_condition(alpha=alpha)
        wing2.set_condition(alpha=alpha)

        # Solve
        wing1.solve()
        wing2.solve()
        CL1[i] = wing1.CL
        CD1[i] = wing1.CD_i
        CL2[i] = wing2.CL
        CD2[i] = wing2.CD_i

    # Plot
    plt.figure()
    plt.plot(CL1, CD1, 'k--', label='No Washout')
    plt.plot(CL2, CD2, 'k-', label='Linear Washout')
    plt.xlabel("CL")
    plt.ylabel("CDi")
    plt.title("Tapered Planform")
    plt.legend()
    plt.show()