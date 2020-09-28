import sys
import json
import scipy

import numpy as np
import matplotlib.pyplot as plt

from abc import abstractmethod


class ObjectInPotentialFlow:
    """An object in a 2D potential flow.

    Parameters
    ----------
    x_le : float
        x coordinate of the leading edge.

    x_te : float
        x coordinate of the trailing edge.
    """

    def __init__(self, **kwargs):
        
        # Get params
        self._x_le = kwargs.get("x_le", 0.0)
        self._x_te = kwargs.get("x_te", 1.0)


    def set_condition(self, **kwargs):
        """Sets the operating condition for the object.

        Parameters
        ----------
        freestream_velocity : float
            Freestream velocity.

        angle_of_attack[deg] : float
            Angle of attack in degrees.

        vortex_strength : float, optional
            Strength of the vortex system.
        """
        
        # Set params
        self._V = kwargs.get("freestream_velocity")
        self._alpha = np.radians(kwargs.get("angle_of_attack[deg]"))
        gamma = kwargs.get("vortex_strength", None)
        if gamma is not None:
            self._gamma = gamma
        elif not hasattr(self, "_gamma"):
            self._gamma = 0.0


    def _get_streamline_derivs(self, point):
        # Returns the derivatives of the streamline with respect to ds at the given point.

        # Get velocity
        v,_ = self._velocity(point)
        V = np.linalg.norm(v)
        return v/V


    def get_streamline(self, start, ds, x_lims):
        """Returns an array of points along a streamline in a two-dimensional velocity
        field defined by velocity. velocity(point) should be defined externally.

        Parameters
        ----------
        start : ndarray
            x, y coordinates of the starting point for the streamline.

        ds : float
            Step length along the streamline for the integration.

        x_lims : list
            Limits in x at which the integration should stop.

        Returns
        -------
        ndarray
            A two-dimensional array of points along the streamline.
        """

        # Determine max iterations (number of steps it would take to go around the outline 4 times)
        max_iterations = abs(int(16*(x_lims[1]-x_lims[0])/ds))

        # Initialize storage
        points = [np.array(start)]

        # Loop
        iterations = 0
        while points[-1][0]>=x_lims[0] and points[-1][0]<=x_lims[1] and iterations<max_iterations:
            iterations += 1

            # Get RK constants
            k1 = self._get_streamline_derivs(points[-1])
            k2 = self._get_streamline_derivs(points[-1]+0.5*ds*k1)
            k3 = self._get_streamline_derivs(points[-1]+0.5*ds*k2)
            k4 = self._get_streamline_derivs(points[-1]+ds*k3)

            # Integrate
            new = points[-1]+0.166666666666666666666666*(k1+2.0*(k2+k3)+k4)*ds
            points.append(new)

        return np.array(points)


    def _surface_normal(self, x):
        # Returns the surface normal vectors on the upper and lower surfaces of the object.

        # Get tangent vectors
        T_u, T_l = self._surface_tangent(x)
        
        # Rotate
        return np.array([-T_u[1], T_u[0]]), np.array([T_l[1], -T_l[0]])


    def _surface_tangent(self, x):
        # Returns the surface tangent vectors on the upper and lower surfaces of the object.
        # These vectors point from the leading edge back

        dx = 1e-6

        # Get points near leading edge
        if abs(x-self._x_le) < dx:
            _, p_u0, p_l0 = self._geometry(x)
            _, p_u1, p_l1 = self._geometry(x+dx)

        # Get points near trailing edge
        elif abs(x-self._x_te) < dx:
            _, p_u0, p_l0 = self._geometry(x-dx)
            _, p_u1, p_l1 = self._geometry(x)

        # Get points elsewhere
        else:
            _, p_u0, p_l0 = self._geometry(x-dx)
            _, p_u1, p_l1 = self._geometry(x+dx)

        T_u = p_u1-p_u0
        T_l = p_l1-p_l0
        T_u = T_u/np.linalg.norm(T_u)
        T_l = T_l/np.linalg.norm(T_l)
        return T_u, T_l


    def _surface_tangential_velocity(self, x):
        # Returns the surface tangential velocity at the x location

        # Get location and tangent vectors
        _, p_u, p_l = self._geometry(x)
        T_u, T_l = self._surface_tangent(x)
        N_u, N_l = self._surface_normal(x)

        # Get velocity stepped off slightly from the surface
        V_u,_ = self._velocity(p_u+N_u*1e-6)
        V_l,_ = self._velocity(p_l+N_l*1e-6)

        # Get tangential velocity
        V_T_u = np.inner(T_u, V_u)
        V_T_l = np.inner(T_l, V_l)
        return V_T_u, V_T_l


    def _stagnation(self):
        # Determines the forward and aft stagnation points

        # Get tangential velocities at various stations to find starting points
        N = 10
        while True:

            # Initialize search
            x = np.zeros(2*N-2)
            x[:N] = np.linspace(self._x_le, self._x_te, N)
            x[N:] = x[N-2:0:-1]
            V_T = np.zeros(2*N-2)

            # Store tangential velocities
            for i in range(2*N-2):
                if i < N:
                    V_T[i],_ = self._surface_tangential_velocity(x[i])
                else:
                    _,V = self._surface_tangential_velocity(x[i])
                    V_T[i] = -V

            # Determine sign changes in the tangential velocity
            signs = np.sign(V_T)
            sz = signs == 0
            while sz.any():
                signs[sz] = np.roll(signs, 1)[sz]
                sz = signs == 0
            sign_changes = ((np.roll(signs, 1) - signs) != 0).astype(int)
            N_sign_changes = np.sum(sign_changes).item()

            # Make sure we have exactly two sign changes
            if N_sign_changes == 2:
                break
            else:
                N *= 2 # Refine search

        # Get indices of sign changes
        chng_locs = np.array([i for i in range(2*N-2)])[np.where(sign_changes)]

        # Get stagnation x locations
        x_stag = [0.0, 0.0]
        for i, ind in enumerate(chng_locs):

            # Sign change on upper surface
            if ind > 0 and ind < N+1:
                x_stag[i] = self._find_stagnation_on_surface(x[ind-1], x[ind], True)
            else:
                x_stag[i] = self._find_stagnation_on_surface(x[ind-1], x[ind], False)

        # Get points
        stag_points = []
        for i, ind in enumerate(chng_locs):

            # Sign change on upper surface
            if ind > 0 and ind < N+1:
                stag_points.append(self._geometry(x_stag[i])[1])
            else:
                stag_points.append(self._geometry(x_stag[i])[2])

        # Sort in x
        if stag_points[1][0] < stag_points[0][0]:
            t = stag_points[1]
            stag_points[1] = stag_points[0]
            stag_points[0] = t

        # Force aft stagnation point to be at the trailing edge for airfoils
        if isinstance(self, VortexPanelAirfoil):
            stag_points[1] = np.array([self._x_te, 0.0])

        return stag_points[0], stag_points[1]

        
    def _find_stagnation_on_surface(self, x0, x1, upper):
        # Finds a stagnation point on a surface using the secant method

        # Get initial guess
        if upper:
            V0,_ = self._surface_tangential_velocity(x0)
            V1,_ = self._surface_tangential_velocity(x1)
        else:
            _,V0 = self._surface_tangential_velocity(x0)
            _,V1 = self._surface_tangential_velocity(x1)

        # Iterate
        e = 1e-10
        e_approx = 1
        while e < e_approx:

            # Determine new guess in x
            x2 = x1-V1*(x1-x0)/(V1-V0)

            # Check bounds
            if x2 < self._x_le:
                x2 = self._x_le+0.001
            if x2 > self._x_te:
                x2 = self._x_te-0.001

            # Get new velocity value
            if upper:
                V2,_ = self._surface_tangential_velocity(x2)
            else:
                _,V2 = self._surface_tangential_velocity(x2)

            # Approximate error
            e_approx = abs(x2-x1)

            # Update for next iteration
            x0 = x1
            V0 = V1
            x1 = x2
            V1 = V2

        return x1


    def plot(self, x_start, x_lims, ds, n, dy):
        """Plots the object in the flow.

        Parameters
        ----------
        x_start : float
            The x location of where the vertical spacing of streamlines should be determined
        
        x_lims : list
            The limits in x for plotting.

        ds : float
            Step size for integrating the streamlines.

        n : float
            Number of streamlines above and below the stagnation streamline.

        dy : float
            Spacing in y of the streamlines.
        """

        # Initialize plot
        plt.figure()

        # Plot geometry
        x_space = np.linspace(self._x_le, self._x_te, 1000)
        camber, upper, lower = self._geometry(x_space)
        plt.plot(camber[:,0], camber[:,1], 'r')
        plt.plot(upper[:,0], upper[:,1], 'b')
        plt.plot(lower[:,0], lower[:,1], 'b')
        plt.xlim(x_lims)
        plt.ylim(x_lims)
        plt.gca().set_aspect('equal', adjustable='box')

        # Determine stagnation points
        print("Locating stagnation points...", end='')
        stag_fwd, stag_bwd = self._stagnation()
        print("Done")

        # Plot stagnation streamlines
        print("Plotting stagnation streamlines...", end='')
        S_stag_fwd = self.get_streamline(stag_fwd-np.array([0.0001,0.0]), -0.01, x_lims)
        S_stag_bwd = self.get_streamline(stag_bwd+np.array([0.0001,0.0]), 0.01, x_lims)
        plt.plot(S_stag_fwd[:,0], S_stag_fwd[:,1], 'k-')
        plt.plot(S_stag_bwd[:,0], S_stag_bwd[:,1], 'k-')
        print("Done")

        # Plot other streamlines
        print("Plotting all streamlines...", end='')
        y_start = np.interp(x_start, S_stag_fwd[:,0], S_stag_fwd[:,1])
        start_point = np.array([x_start, y_start])
        for i in range(n):
            point = np.copy(start_point)
            point[1] += dy*(i+1)
            streamline = self.get_streamline(point, 0.01, x_lims)
            plt.plot(streamline[:,0], streamline[:,1], 'k-')
        for i in range(n):
            point = np.copy(start_point)
            point[1] -= dy*(i+1)
            streamline = self.get_streamline(point, 0.01, x_lims)
            plt.plot(streamline[:,0], streamline[:,1], 'k-')
        print("Done")

        # Format and show plot
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    
    @abstractmethod
    def _velocity(self, point):
        pass

    @abstractmethod
    def _geometry(self, x):
        pass


class VortexPanelAirfoil(ObjectInPotentialFlow):
    """An airfoil characterized using vortex panel method.

    Parameters
    ----------
    NACA : str
        4-digit NACA designation of the airfoil.

    x_le : float
        x coordinate of the leading edge.

    x_te : float
        x coordinate of the trailing edge.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set params
        self._NACA = kwargs.get("NACA", "0012")
        self._c = self._x_te-self._x_le


    def _velocity(self, point):
        # Returns the velocity components and coefficient of pressure at the Cartesian coordinates given

        # Determine chi-eta coordinates of the point wrt each panel
        dxy = point[np.newaxis,:]-self._p_N[:-1,:]
        chi_eta = np.matmul(self._T, dxy[:,:,np.newaxis])
        chi = chi_eta[:,0,0].flatten()
        eta = chi_eta[:,1,0].flatten()

        # Calculate phi and psi for the point from each panel
        E_2_n_2 = eta**2+chi**2
        phi = np.arctan2(eta*self._l, E_2_n_2-chi*self._l)
        psi = 0.5*np.log(E_2_n_2/((chi-self._l)**2+eta**2))

        # Calculate the influence matrices
        G = np.ones((self._N, 2, 2))/(2.0*np.pi*self._l[:,np.newaxis,np.newaxis])
        G[:,0,0] *= (self._l-chi)*phi+eta*psi
        G[:,0,1] *= chi*phi-eta*psi
        G[:,1,0] *= eta*phi-(self._l-chi)*psi-self._l
        G[:,1,1] *= -eta*phi-chi*psi+self._l
        P = np.matmul(np.transpose(self._T, (0,2,1)), G)

        # Sum velocity
        V = np.ones(2)*self._V
        V[0] *= np.cos(self._alpha)
        V[1] *= np.sin(self._alpha)
        for i in range(self._N):
            V += np.matmul(P[i], self._gamma[i:i+2,np.newaxis]).flatten()

        V_mag = np.linalg.norm(V)

        return V, 1.0-(V_mag/self._V)**2


    def panel(self, N):
        """Discretizes the airfoil surface into panels for numerically solving the flow.
        This function also serves to generate the influence matrix.

        Parameters
        ---------
        N : int
            Number of panels.
        """

        # Store number of panels
        self._N = N
        n = N+1 # Number of nodes

        # Initialize gamma array
        self._gamma = np.zeros(n)

        # Generate even number of nodes
        if n%2 == 0:
            d_theta = np.pi/((n/2)-0.5)
            theta = np.linspace(0.5*d_theta, np.pi, int(n/2))
            x = 0.5*(1.0-np.cos(theta))*self._c

        # Generate odd number of nodes
        else:
            theta = np.linspace(0.0, np.pi, int(n/2)+1)
            x = 0.5*(1.0-np.cos(theta))*self._c

        # Get raw outline points
        _, p_upper, p_lower = self._geometry(x)

        # Initialize node array
        self._p_N = np.zeros((n, 2))

        # Organize nodes
        if n%2 == 0:
            self._p_N[:int(n/2),:] = p_lower[::-1,:]
            self._p_N[int(n/2):,:] = p_upper
        else:
            self._p_N[:int(n/2)+1,:] = p_lower[::-1,:]
            self._p_N[int(n/2)+1:,:] = p_upper[1:,:]

        # Determine control points and panel lengths
        self._p_C = 0.5*(self._p_N[:-1,:]+self._p_N[1:,:])
        self._l = np.linalg.norm(self._p_N[:-1,:]-self._p_N[1:,:], axis=1)

        # Determine transformation matrix from x-y to chi-eta
        self._T = np.ones((self._N,2,2))/self._l[:,np.newaxis,np.newaxis]
        self._T[:,0,0] *= self._p_N[1:,0]-self._p_N[:-1,0]
        self._T[:,0,1] *= self._p_N[1:,1]-self._p_N[:-1,1]
        self._T[:,1,0] *= -(self._p_N[1:,1]-self._p_N[:-1,1])
        self._T[:,1,1] *= self._p_N[1:,0]-self._p_N[:-1,0]

        # Determine chi-eta coordinates of each control point; first index is the panel, second index is the control point
        dxy = self._p_C[np.newaxis,:,:]-self._p_N[:-1,np.newaxis,:]
        chi_eta = np.matmul(self._T[:,np.newaxis,:,:], dxy[:,:,:,np.newaxis])
        chi = chi_eta[:,:,0,0]
        eta = chi_eta[:,:,1,0]

        # Calculate phi and psi for the control points
        E_2_n_2 = eta**2+chi**2
        phi = np.arctan2(eta*self._l[:,np.newaxis], E_2_n_2-chi*self._l[:,np.newaxis])
        psi = 0.5*np.log(E_2_n_2/((chi-self._l[:,np.newaxis])**2+eta**2))

        # Calculate the influence matrices
        G = np.ones((self._N, self._N, 2, 2))/(2.0*np.pi*self._l[:,np.newaxis,np.newaxis,np.newaxis])
        G[:,:,0,0] *= (self._l[:,np.newaxis]-chi)*phi+eta*psi
        G[:,:,0,1] *= chi*phi-eta*psi
        G[:,:,1,0] *= eta*phi-(self._l[:,np.newaxis]-chi)*psi-self._l[:,np.newaxis]
        G[:,:,1,1] *= -eta*phi-chi*psi+self._l[:,np.newaxis]
        P = np.matmul(np.transpose(self._T, (0,2,1))[:,np.newaxis,:,:], G)

        # Determine A matrix
        self._A = np.zeros((n, n))
        for i in range(self._N):
            for j in range(self._N):
                self._A[i,j] += ((self._p_N[i+1,0]-self._p_N[i,0])*P[j,i,1,0]-(self._p_N[i+1,1]-self._p_N[i,1])*P[j,i,0,0])/self._l[i]
                self._A[i,j+1] += ((self._p_N[i+1,0]-self._p_N[i,0])*P[j,i,1,1]-(self._p_N[i+1,1]-self._p_N[i,1])*P[j,i,0,1])/self._l[i]

        # Kutta condition
        self._A[-1,0] = 1.0
        self._A[-1,-1] = 1.0

        # Perform LU decomposition


    def set_condition(self, **kwargs):
        """Specify the given condition. This function serves to generate the b vector.

        Parameters
        ----------
        alpha : float
            Angle of attack in degrees.

        V : float
            Freestream velocity.
        """

        # Store angle of attack
        self._alpha = np.radians(kwargs["alpha"])
        self._V = kwargs["V"]
        C_a = np.cos(self._alpha)
        S_a = np.sin(self._alpha)

        # Populate b vector
        self._b = np.zeros(self._N+1)
        self._b[:-1] = self._V*((self._p_N[1:,1]-self._p_N[:-1,1])*C_a-(self._p_N[1:,0]-self._p_N[:-1,0])*S_a)/self._l


    def solve(self):
        """Solve the airfoil at the current condition with the current panelling.
        """

        # Solve matrix system
        self._gamma = np.linalg.solve(self._A, self._b)

        # Determine CL
        CL = np.sum(self._l*(self._gamma[:-1]+self._gamma[1:])/(self._V*self._c))

        # Determine Cm_le
        C_a = np.cos(self._alpha)
        S_a = np.sin(self._alpha)
        integrand = self._l*(C_a*(2.0*self._p_N[:-1,0]*self._gamma[:-1]
                                  +self._p_N[:-1,0]*self._gamma[1:]
                                  +self._p_N[1:,0]*self._gamma[:-1]
                                  +2.0*self._p_N[1:,0]*self._gamma[1:])
                             +S_a*(2.0*self._p_N[:-1,1]*self._gamma[:-1]
                                   +self._p_N[:-1,1]*self._gamma[1:]
                                   +self._p_N[1:,1]*self._gamma[:-1]
                                   +2.0*self._p_N[1:,1]*self._gamma[1:]))
        Cm_le = -1.0/(3.0*self._c**2*self._V)*np.sum(integrand)

        Cm_c4 = Cm_le+0.25*CL*C_a

        return CL, Cm_le, Cm_c4


    def _geometry(self, x):
        # Calculates the geometry
        self._m = float(self._NACA[0])/100
        self._p = float(self._NACA[1])/10
        self._t = float(self._NACA[2:])/100
        x_c = x/self._c

        # Camber line
        if self._p != 0.0:
            y_c =  np.where(x_c<self._p, self._m/(self._p*self._p)*(2*self._p*x_c-x_c*x_c), self._m/((1-self._p)*(1-self._p))*(1-2*self._p+2*self._p*x_c-x_c*x_c))
        else:
            y_c =  np.zeros_like(x_c)

        # Determine camber line derivative
        if abs(self._m)<1e-10 or abs(self._p)<1e-10: # Symmetric
            dy_c_dx = np.zeros_like(x_c)
        else:
            dy_c_dx = np.where(x_c<self._p, 2*self._m/(self._p*self._p)*(self._p-x_c), 2*self._m/((1-self._p)*(1-self._p))*(self._p-x_c))

        # Thickness
        t =  5.0*self._t*(0.2969*np.sqrt(x_c)-0.1260*x_c-0.3516*x_c*x_c+0.2843*x_c*x_c*x_c-0.1015*x_c*x_c*x_c*x_c)

        # Outline points
        x_upper = (x_c-t*np.sin(np.arctan(dy_c_dx)))*self._c
        y_upper = (y_c+t*np.cos(np.arctan(dy_c_dx)))*self._c
        x_lower = (x_c+t*np.sin(np.arctan(dy_c_dx)))*self._c
        y_lower = (y_c-t*np.cos(np.arctan(dy_c_dx)))*self._c

        return np.array([x, y_c*self._c]).T, np.array([x_upper, y_upper]).T, np.array([x_lower, y_lower]).T


if __name__=="__main__":

    # Get input
    input_file = sys.argv[-1]
    with open(input_file, 'r') as input_handle:
        input_dict = json.load(input_handle)

    # Initialize object
    geom_dict = input_dict["geometry"]
    airfoil = VortexPanelAirfoil(NACA=geom_dict["NACA"],
                                 x_le=geom_dict["x_leading_edge"],
                                 x_te=geom_dict["x_trailing_edge"])

    # Initialize panels
    airfoil.panel(geom_dict["N"])

    # Get iteration params
    oper_dict = input_dict["operating"]
    V = oper_dict["freestream_velocity"]
    a_start = oper_dict["alpha_start[deg]"]
    a_end = oper_dict["alpha_end[deg]"]
    a_inc = oper_dict["alpha_increment"]
    n = int((a_end-a_start)/a_inc+1)

    # Start table
    print("".join(["-"]*80))
    print("NACA {0}".format(airfoil._NACA))
    print("".join(["-"]*9))
    print("{:<20}{:<20}{:<20}{:<20}".format("Alpha [deg]", "CL", "Cm_le", "Cm_c4"))
    print("".join(["-"]*80))

    # Iterate
    for a in np.linspace(a_start, a_end, n):

        # Set condition
        airfoil.set_condition(alpha=a, V=V)

        if a == 0.0:
            print(airfoil._p_N)

        # Solve
        coefs = airfoil.solve()
        print("{:<20.6}{:<20.6}{:<20.6}{:<20.6}".format(a, *coefs))

    print("".join(["-"]*80))

    ## Plot
    #plot_dict = input_dict["plot"]
    #airfoil.plot(plot_dict["x_start"],
    #             [plot_dict["x_lower_limit"], plot_dict["x_upper_limit"]],
    #             plot_dict["delta_s"],
    #             plot_dict["n_lines"],
    #             plot_dict["delta_y"])