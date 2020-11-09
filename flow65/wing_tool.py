import sys
import json

import numpy as np

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
        self._CL_a = np.pi*self._AR*self._a_n[0]

        return self._CL_a


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
        self._CL = np.pi*self._AR*self._A_n[0]

        # Determine drag coefficient
        self._CD_i = np.pi*self._AR*np.sum(self._N_range*self._A_n**2)

        return self._CL, self._CD_i


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
    CLa = wing.set_grid(wing_dict["nodes_per_semispan"])

    # Set condition
    wing.set_condition(alpha=input_dict["condition"]["alpha_root[deg]"])

    # Solve
    CL, CD_i = wing.solve()
    print("CL: {0}".format(CL))
    print("CD_i: {0}".format(CD_i))
    print("CL,a: {0}".format(CLa))