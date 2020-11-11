import sys
import json

import numpy as np
import matplotlib.pyplot as plt


class Wing:
    """A class for modeling a finite wing using the sine-series solution to Prandtl's lifting-line equation.

    Parameters
    ----------
    planform : str
        May be "elliptic" or "tapered".

    AR : float
        Aspect ratio.

    RT : float
        Taper ratio. Only required for "tapered" planform.

    CL_a_section : float, optional
        Section lift slope. Defaults to 2 pi.
    """

    def __init__(self, **kwargs):

        # Get parameters
        self._planform_type = kwargs["planform"]
        self._AR = kwargs["AR"]
        if self._planform_type == "tapered":
            self._RT = kwargs["RT"]
        self._CL_a_s = kwargs.get("CL_a_section", 2.0*np.pi)


    def set_grid(self, N):
        """Sets the spanwise grid for the wing. Uses cosine clustering

        Parameters
        ----------
        N : int
            Number of nodes per semispan to specify. Note that one node will
            be placed at the root, making the total number of nodes 2N-1.
        """

        # Create theta and z distributions
        self._N = 2*N-1
        self._theta = np.linspace(0, np.pi, self._N)
        self._z = -0.5*np.cos(self._theta)

        # Calculate control point trig values
        self._N_range = np.arange(1, self._N+1)
        self._S_theta = np.sin(self._theta)

        # Calculate chord values
        if self._planform_type == "elliptic":
            self._c_b = 2.0*self._S_theta/(np.pi*self._AR)
        else:
            self._c_b = 2.0*(1.0-(1.0-self._RT)*np.abs(np.cos(self._theta)))/(self._AR*(1.0+self._RT))

        self._c_b = np.where(self._c_b==0.0, 1e-6, self._c_b)

        # Get C matrix
        self._C = np.zeros((self._N, self._N))
        self._C[0,:] = self._N_range**2
        self._C[1:-1,:] = (4.0/(self._CL_a_s*self._c_b[1:-1,np.newaxis])+self._N_range[np.newaxis,:]/self._S_theta[1:-1,np.newaxis])*np.sin(self._N_range[np.newaxis,:]*self._theta[1:-1,np.newaxis])
        self._C[-1,:] = (-1.0)**(self._N_range+1)*self._N_range**2

        # Get C inverse (why on earth, I have no idea...)
        self._C_inv = np.linalg.inv(self._C)

        np.set_printoptions(linewidth=np.inf, precision=12)

        # Determine the Fourier coefficients
        self._a_n = np.linalg.solve(self._C, np.ones(self._N))

        # Determine coefficient slopes
        self.CL_a = np.pi*self._AR*self._a_n[0]

        # Determine the kappa factors
        self.K_D = np.sum(np.arange(2, self._N+1)*self._a_n[1:]**2/self._a_n[0]**2)
        A = (1+np.pi*self._AR/self._CL_a_s)*self._a_n[0]
        self.K_L = (1.0-A)/A

        # Determine span efficiency factor
        self.e_s = 1.0/(1.0+self.K_D)


    def set_condition(self, **kwargs):
        """Sets atmospheric condition for the wing.

        Parameters
        ----------
        alpha : float
            Angle of attack in degrees.
        """

        # Get angle of attack
        self._alpha = np.radians(kwargs["alpha"])


    def solve(self):
        """Solves for the aerodynamic coefficients at the current condition."""

        # Determine Fourier coefficients dependent on condition
        self._A_n = self._a_n*(self._alpha)

        # Determine lift coefficient
        self.CL = np.pi*self._AR*self._A_n[0]

        # Determine drag coefficient
        self.CD_i = np.pi*self._AR*np.sum(self._N_range*self._A_n**2)


    def plot_planform(self):
        """Shows a plot of the planform."""

        # Get leading and trailing edge points
        x_le = np.zeros(self._N+2)
        x_te = np.zeros(self._N+2)
        x_le[1:-1] = 0.25*self._c_b
        x_te[1:-1] = -0.75*self._c_b
        z = np.zeros(self._N+2)
        z[0] = self._z[0]
        z[1:-1] = self._z
        z[-1] = self._z[-1]

        # Plot
        plt.figure()
        plt.plot(z, x_le, 'k-')
        plt.plot(z, x_te, 'k-')
        plt.plot(z, np.zeros(self._N+2), 'b-', label='c/4')
        plt.xlabel('z/b')
        plt.ylabel('x/b')
        plt.title('Planform')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend(loc='upper right')
        plt.show()


if __name__=="__main__":

    # Read in input
    input_file = sys.argv[-1]
    with open(input_file, 'r') as input_handle:
        input_dict = json.load(input_handle)

    # Initialize wing
    wing_dict = input_dict["wing"]
    wing = Wing(planform=wing_dict["planform"]["type"],
                AR=wing_dict["planform"]["aspect_ratio"],
                RT=wing_dict["planform"].get("taper_ratio"),
                CL_a_section=wing_dict["airfoil_lift_slope"])
    
    # Set up grid
    wing.set_grid(wing_dict["nodes_per_semispan"])

    # Set condition
    wing.set_condition(alpha=input_dict["condition"]["alpha_root[deg]"])

    # Solve
    wing.solve()

    print()
    print("Wing")
    print("    Type: {0}".format(wing._planform_type))
    print("    Aspect Ratio: {0}".format(wing._AR))
    try:
        print("    Taper Ratio: {0}".format(wing._RT))
    except AttributeError:
        pass

    print()
    print("Condition")
    print("    Alpha: {0} deg".format(np.degrees(wing._alpha)))

    print()
    print("Aerodynamic Coefficients")
    print("    CL: {0}".format(wing.CL))
    print("    CD_i: {0}".format(wing.CD_i))

    print()
    print("Performance Parameters")
    print("    CL,a: {0}".format(wing.CL_a))
    print("    K_L: {0}".format(wing.K_L))
    print("    K_D: {0}".format(wing.K_D))
    print("    Span Efficiency: {0}".format(wing.e_s))

    # Check for plot request
    if input_dict["view"]["planform"]:
        wing.plot_planform()