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

    washout : str
        May be "none", "linear", or "optimum".

    washout_mag : float
        Magnitude of the washout at the tip in degrees.

    washout_CLd : float
        Design lift coefficient for washout. Only required if "washout"
        is "optimum".

    aileron_lims : list
        Aileron limits as a fraction of the span.

    aileron_cf: list
        Aileron chord fractions at the root and tip of the ailerons.

    aileron_hinge_eff : float
        Aileron hinge efficiency.
    """

    def __init__(self, **kwargs):

        # Get planform parameters
        self._planform_type = kwargs["planform"]
        self._AR = kwargs["AR"]
        if self._planform_type == "tapered":
            self._RT = kwargs["RT"]
        self._CL_a_s = kwargs.get("CL_a_section", 2.0*np.pi)

        # Get washout parameters
        self._washout_type = kwargs.get("washout", "none")
        if self._washout_type != "none":
            self._W = np.radians(kwargs.get("washout_mag", 0.0))
        else:
            self._W = 0.0
        
        if self._washout_type == "optimum":
            self._CL_d = kwargs["washout_CLd"]

        # Get aileron parameters
        self._aln_lims = kwargs["aileron_lims"]
        self._aln_cf = kwargs["aileron_cf"]
        self._aln_e_hinge = kwargs["aileron_hinge_eff"]


    def set_grid(self, N):
        """Sets the spanwise grid for the wing. Uses cosine clustering

        Parameters
        ----------
        N : int
            Number of nodes per semispan to specify. Note that one node will
            be placed at the root, making the total number of nodes 2N-1.
        """

        np.set_printoptions(linewidth=np.inf, precision=12)

        # Create theta and z distributions
        self._N = 2*N-1
        self._theta = np.linspace(0, np.pi, self._N)
        self._z = -0.5*np.cos(self._theta)

        # Calculate control point trig values
        self._N_range = np.arange(1, self._N+1)
        self._S_theta = np.sin(self._theta)

        # Calculate chord distribution
        if self._planform_type == "elliptic":
            self._c_b = 4.0*self._S_theta/(np.pi*self._AR)
        else:
            self._c_b = 2.0*(1.0-(1.0-self._RT)*np.abs(np.cos(self._theta)))/(self._AR*(1.0+self._RT))
        self._c_b = np.where(self._c_b==0.0, 1e-6, self._c_b)

        # Calculate washout distribution
        if self._washout_type == "none":
            self._w = np.zeros(self._N)
        elif self._washout_type == "linear":
            self._w = np.abs(np.cos(self._theta))
        elif self._washout_type == "optimum":
            self._w = 1.0-self._S_theta*self._c_b[self._N//2]/self._c_b
            self._W = 4.0*self._CL_d/(np.pi*self._AR*self._CL_a_s*self._c_b[self._N//2])

        # Determine aileron chord fractions
        self._cf = np.zeros(self._N)
        z_in_aileron = ((self._z>self._aln_lims[0]) & (self._z<self._aln_lims[1])) | ((self._z>-self._aln_lims[1]) & (self._z<-self._aln_lims[0]))
        if self._planform_type == "elliptic":
            self._x_h_tip = -(1.0-self._aln_cf[1]-0.25)*(4.0/(np.pi*self._AR)*np.sqrt(1.0-(2.0*self._aln_lims[1])**2))
            self._x_h_root = -(1.0-self._aln_cf[0]-0.25)*(4.0/(np.pi*self._AR)*np.sqrt(1.0-(2.0*self._aln_lims[0])**2))
        else:
            self._x_h_tip = -(1.0-self._aln_cf[1]-0.25)*(2.0/(self._AR*(1.0+self._RT))*(1.0-(1.0-self._RT)*2.0*self._aln_lims[1]))
            self._x_h_root = -(1.0-self._aln_cf[0]-0.25)*(2.0/(self._AR*(1.0+self._RT))*(1.0-(1.0-self._RT)*2.0*self._aln_lims[0]))

        aln_b = (self._x_h_tip-self._x_h_root)/(self._aln_lims[1]-self._aln_lims[0])
        x_h = z_in_aileron[self._N//2:]*(self._x_h_root+(self._z[self._N//2:]-self._aln_lims[0])*aln_b)
        self._cf[self._N//2:] = 1.0-(-x_h/self._c_b[self._N//2:]+0.25)
        self._cf[self._N//2::-1] = 1.0-(-x_h/self._c_b[self._N//2:]+0.25)
        self._cf *= z_in_aileron

        # Determine flap efficiency
        theta_f = np.arccos(2.0*self._cf-1.0)
        self._e_f = (1.0-(theta_f-np.sin(theta_f))/np.pi)*self._aln_e_hinge
        self._e_f[self._N//2:] *= -1.0

        # Get C matrix
        self._C = np.zeros((self._N, self._N))
        self._C[0,:] = self._N_range**2
        self._C[1:-1,:] = (4.0/(self._CL_a_s*self._c_b[1:-1,np.newaxis])+self._N_range[np.newaxis,:]/self._S_theta[1:-1,np.newaxis])*np.sin(self._N_range[np.newaxis,:]*self._theta[1:-1,np.newaxis])
        self._C[-1,:] = (-1.0)**(self._N_range+1)*self._N_range**2

        # Get C inverse (why on earth, I have no idea...)
        self._C_inv = np.linalg.inv(self._C)

        # Determine the Fourier coefficients
        self._a_n = np.linalg.solve(self._C, np.ones(self._N))
        self._b_n = np.linalg.solve(self._C, self._w)
        self._c_n = np.linalg.solve(self._C, self._e_f)
        self._d_n = np.linalg.solve(self._C, np.cos(self._theta))

        # Determine coefficient slopes
        self.CL_a = np.pi*self._AR*self._a_n[0]

        # Determine the kappa factors due to planform
        self.K_D = np.sum(np.arange(2, self._N+1)*self._a_n[1:]**2/self._a_n[0]**2)
        A = (1+np.pi*self._AR/self._CL_a_s)*self._a_n[0]
        self.K_L = (1.0-A)/A

        # Determine span efficiency factor
        self.e_s = 1.0/(1.0+self.K_D)

        # Determine kappa factors due to washout
        if self._washout_type != "none":
            self.e_omega = self._b_n[0]/self._a_n[0]
            self.K_DL = 2.0*self._b_n[0]/self._a_n[0]*np.sum(self._N_range[1:]*self._a_n[1:]/self._a_n[0]*(self._b_n[1:]/self._b_n[0]-self._a_n[1:]/self._a_n[0]))
            self.K_Domega = (self._b_n[0]/self._a_n[0])**2*np.sum(self._N_range[1:]*(self._b_n[1:]/self._b_n[0]-self._a_n[1:]/self._a_n[0])**2)
            self.K_Do = self.K_D-0.25*self.K_DL**2/self.K_Domega
        else:
            self.e_omega = 0.0
            self.K_DL = 0.0
            self.K_Domega = 0.0
            self.K_Do = 0.0

        # Determine aileron and roll derivatives
        self.Cl_da = -0.25*np.pi*self._AR*self._c_n[1]
        self.Cl_p = -0.25*np.pi*self._AR*self._d_n[1]
        print(self._C)
        print(self._d_n)


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
        self._A_n = self._a_n*(self._alpha)-self._b_n*self._W

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

        # Plot outline and LQC
        plt.figure()
        plt.plot(z, x_le, 'k-')
        plt.plot(z, x_te, 'k-')
        plt.plot(z, np.zeros(self._N+2), 'b-', label='c/4')

        # Plot spanwise stations
        for i in range(self._N):
            plt.plot([z[i+1], z[i+1]], [x_le[i+1], x_te[i+1]], 'b--')

        # Plot ailerons
        plt.plot(self._aln_lims, [self._x_h_root, self._x_h_tip], 'k-')
        plt.plot([-self._aln_lims[0], -self._aln_lims[1]], [self._x_h_root, self._x_h_tip], 'k-')

        # Plot labels
        plt.xlabel('z/b')
        plt.ylabel('x/b')
        plt.title('Planform')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend(loc='upper right')
        plt.show()


    def plot_washout(self):
        """Plots the washout distribution on the wing."""

        plt.figure()
        plt.plot(self._z, self._w, 'k-')
        plt.xlabel("z/b")
        plt.ylabel("Washout [deg]")
        plt.title("Washout Distribution")
        plt.show()


if __name__=="__main__":

    # Read in input
    input_file = sys.argv[-1]
    with open(input_file, 'r') as input_handle:
        input_dict = json.load(input_handle)

    # Initialize wing
    wing_dict = input_dict["wing"]
    washout_dict = input_dict["wing"]["washout"]
    aileron_dict = input_dict["wing"]["aileron"]
    wing = Wing(planform=wing_dict["planform"]["type"],
                AR=wing_dict["planform"]["aspect_ratio"],
                RT=wing_dict["planform"].get("taper_ratio"),
                CL_a_section=wing_dict["airfoil_lift_slope"],
                washout=washout_dict["distribution"],
                washout_mag=washout_dict["magnitude[deg]"],
                washout_CLd=washout_dict["CL_design"],
                aileron_lims=[aileron_dict["begin[z/b]"], aileron_dict["end[z/b]"]],
                aileron_cf=[aileron_dict["begin[cf/c]"], aileron_dict["end[cf/c]"]],
                aileron_hinge_eff=aileron_dict["hinge_efficiency"])
    
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
    print("Planform Effects")
    print("    CL,a: {0}".format(wing.CL_a))
    print("    K_L: {0}".format(wing.K_L))
    print("    K_D: {0}".format(wing.K_D))
    print("    Span efficiency: {0}".format(wing.e_s))

    print()
    print("Washout Effects")
    print("    Washout effectiveness: {0}".format(wing.e_omega))
    print("    K_DL: {0}".format(wing.K_DL))
    print("    Washout contribution to induced drag: {0}".format(wing.K_Domega))
    print("    K_Do: {0}".format(wing.K_Do))

    print()
    print("Aileron Effects")
    print("    Cl,da: {0}".format(wing.Cl_da))

    print()
    print("Roll Effects")
    print("    Cl,p: {0}".format(wing.Cl_p))

    # Check for plot requests
    if input_dict["view"]["planform"]:
        wing.plot_planform()
    if input_dict["view"]["washout_distribution"]:
        wing.plot_washout()

    # Write solution
    with open("Solution.txt", 'w') as f:
        C_str = np.array2string(wing._C)
        C_inv_str = np.array2string(wing._C_inv)
        a_n_str = np.array2string(wing._a_n)
        b_n_str = np.array2string(wing._b_n)
        c_n_str = np.array2string(wing._b_n)
        d_n_str = np.array2string(wing._b_n)

        print("C array", file=f)
        print(C_str, file=f)
        print("C_inv array", file=f)
        print(C_inv_str, file=f)
        print("a_n", file=f)
        print(a_n_str, file=f)
        print("b_n", file=f)
        print(b_n_str, file=f)
        print("c_n", file=f)
        print(c_n_str, file=f)
        print("d_n", file=f)
        print(d_n_str, file=f)