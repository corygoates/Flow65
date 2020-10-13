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
        self._c = self._x_te-self._x_le


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
        eps = 1e-3
        V_u,_ = self._velocity(p_u+N_u*eps)
        V_l,_ = self._velocity(p_l+N_l*eps)

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
            x[:N] = np.linspace(self._x_le, self._x_te-1e-10, N)
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


    def plot_streamlines(self, x_start, x_lims, ds, n, dy, plot_stagnation=True):
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

        plot_stagnation : bool, optional
            Whether to plot the stagnation streamlines. Defaults to True.
        """

        # Initialize plot
        plt.figure()

        # Plot geometry
        y_start = 0.0
        if hasattr(self, "_airfoil") and self._airfoil == "file":
            plt.plot(self._p_N[:,0], self._p_N[:,1], 'k')
        else:
            x_space = np.linspace(self._x_le, self._x_te, 1000)
            camber, upper, lower = self._geometry(x_space)
            plt.plot(camber[:,0], camber[:,1], 'r--')
            plt.plot(upper[:,0], upper[:,1], 'k')
            plt.plot(lower[:,0], lower[:,1], 'k')

        # Determine stagnation points
        if plot_stagnation:
            print("Locating stagnation points...", end='', flush=True)
            if hasattr(self, "_airfoil") and self._airfoil == "file":

                # Determine normal vectors at control points
                self._n = np.zeros((self._N, 2))
                self._n[:,0] += -(self._p_N[1:,1]-self._p_N[:-1,1])/self._l
                self._n[:,1] += (self._p_N[1:,0]-self._p_N[:-1,0])/self._l

                # Offset control points
                eps = 1e-3
                V_points = self._p_C+self._n*eps

                # Get V
                V = np.zeros((self._N,2))
                for i, p in enumerate(V_points):
                    V[i],_ = self._velocity(p)

                # Get stagnation points
                stag_fwd = V_points[np.argmin(V[self._N//4:self._N-self._N//4])]
                stag_bwd = [self._x_te, 0.0]

            else:
                stag_fwd, stag_bwd = self._stagnation()
            print("Done")

            # Plot stagnation streamlines
            print("Plotting stagnation streamlines...", end='', flush=True)
            S_stag_fwd = self.get_streamline(stag_fwd-np.array([0.0001,0.0]), -ds, x_lims)
            S_stag_bwd = self.get_streamline(stag_bwd+np.array([0.0001,0.0]), ds, x_lims)
            plt.plot(S_stag_fwd[:,0], S_stag_fwd[:,1], 'b')
            plt.plot(S_stag_bwd[:,0], S_stag_bwd[:,1], 'b')
            print("Done")
            y_start = np.interp(x_start, S_stag_fwd[:,0], S_stag_fwd[:,1])

        # Plot other streamlines
        print("Plotting all streamlines...", end='', flush=True)
        start_point = np.array([x_start, y_start])
        for i in range(n):
            point = np.copy(start_point)
            point[1] += dy*(i+1)
            streamline = self.get_streamline(point, ds, x_lims)
            plt.plot(streamline[:,0], streamline[:,1], 'b')
        for i in range(n):
            point = np.copy(start_point)
            point[1] -= dy*(i+1)
            streamline = self.get_streamline(point, ds, x_lims)
            plt.plot(streamline[:,0], streamline[:,1], 'b')
        print("Done")

        # Format and show plot
        plt.ylim([y_start-dy*(n+1), y_start+dy*(n+1)])
        plt.xlim(x_lims)
        plt.gca().set_aspect('equal', adjustable='box')
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
    airfoil : str
        4-digit NACA designation of the airfoil or "ULxx" to specify a
        uniform load airfoil with a NACA thickness distribution with
        maximum thickness of xx. Defaults to "0012".

    CL_design : float
        Design CL at zero angle of attack for uniform load airfoil.

    trailing_edge : str
        "open" or "closed". Determines the equations for the thickness distribution.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set params
        self._airfoil = kwargs.get("airfoil", "0012")
        if "UL" in self._airfoil:
            self._camber_type = "UL"
            self._CL_d = kwargs["CL_design"]
        elif self._airfoil == "file":
            self._camber_type = "file_defined"
            self._load_geom_from_file(kwargs["filename"])
        else:
            self._camber_type = "NACA"
        self._c = self._x_te-self._x_le
        self._close_te = kwargs.get("trailing_edge") == "closed"

        # Initialize panels
        if self._airfoil != "file":
            self.panel(kwargs["n_points"]-1)


    def _load_geom_from_file(self, filename):
        # Reads in node locations from file

        # Load data
        self._p_N = np.genfromtxt(filename)
        self._N = self._p_N.shape[0]-1

        # Perform paneling
        self.panel(self._N)


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

        n = N+1 # Number of nodes
        if self._camber_type != "file_defined":

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

        elif N != self._N:
            raise RuntimeError("panel() may not be called on a file-defined airfoil when the desired number of panels is different than that given in the file.")

        # Determine control points and panel lengths
        self._N = N
        self._p_C = 0.5*(self._p_N[:-1,:]+self._p_N[1:,:])
        self._l = np.linalg.norm(self._p_N[:-1,:]-self._p_N[1:,:], axis=1)

        # Initialize gamma array
        self._gamma = np.zeros(n)

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


    def plot_C_P(self):
        """Plots the pressure coefficient distribution on the surface of the
        airfoil at the current condition.
        """

        # Determine normal vectors at control points
        self._n = np.zeros((self._N, 2))
        self._n[:,0] += -(self._p_N[1:,1]-self._p_N[:-1,1])/self._l
        self._n[:,1] += (self._p_N[1:,0]-self._p_N[:-1,0])/self._l

        # Offset control points
        eps = 1e-3
        C_P_points = self._p_C+self._n*eps

        # Get C_P
        C_P = np.zeros(self._N)
        for i, p in enumerate(C_P_points):
            _,C_P[i] = self._velocity(p)

        # Plot
        plt.figure()
        plt.plot(self._p_C[:self._N//2,0], C_P[:self._N//2], 'k--', label='Bottom')
        plt.plot(self._p_C[self._N//2:,0], C_P[self._N//2:], 'k', label='Top')
        plt.gca().invert_yaxis()
        plt.xlabel("x/c")
        plt.ylabel("C_P")
        plt.legend()
        plt.show()


    def _geometry(self, x):
        # Calculates the geometry
        x_c = x/self._c
        self._t = float(self._airfoil[2:])/100
        if self._camber_type == "NACA":
            self._m = float(self._airfoil[0])/100
            self._p = float(self._airfoil[1])/10

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

        else:

            # Camber line and derivative; the derivative can be zero at the leading edge because the thickness is zero there.
            with np.errstate(invalid='ignore', divide='ignore'):
                const = self._CL_d*self._c/(4.0*np.pi)
                y_c = const*np.where((x_c != 0.0) & (x_c != 1.0), ((x_c-1.0)*np.log(1.0-x_c)-x_c*np.log(x_c)), 0.0)
                dy_c_dx = const*np.where(x_c != 1.0, np.where(x_c != 0.0, np.log(1.0-x_c)-np.log(x_c), 0.0), np.log(1e-15))


        # Thickness
        if self._close_te:
            t =  0.5*self._t*(2.980*np.sqrt(x_c)-1.320*x_c-3.286*x_c*x_c+2.441*x_c*x_c*x_c-0.815*x_c*x_c*x_c*x_c)
        else:
            t =  0.5*self._t*(2.969*np.sqrt(x_c)-1.260*x_c-3.516*x_c*x_c+2.843*x_c*x_c*x_c-1.015*x_c*x_c*x_c*x_c)

        # Outline points
        theta = np.arctan(dy_c_dx)
        x_upper = (x_c-t*np.sin(theta))*self._c
        y_upper = (y_c+t*np.cos(theta))*self._c
        x_lower = (x_c+t*np.sin(theta))*self._c
        y_lower = (y_c-t*np.cos(theta))*self._c

        return np.array([x, y_c*self._c]).T, np.array([x_upper, y_upper]).T, np.array([x_lower, y_lower]).T


    def export_geometry(self):
        """Exports the current paneling geometry."""

        # Get filename
        filename = self._airfoil+".txt"

        # Export points
        header = "{:<20} {:<20}".format('x', 'y')
        fmt_str = "%20.12e %20.12e"
        np.savetxt(filename, self._p_N, fmt=fmt_str, header=header)


class CylinderInComplexPotentialFlow(ObjectInPotentialFlow):
    """A cylinder in potential flow.

    Parameters
    ----------
    cylinder_radius : float
        Radius of the cylinder.

    z : list
        Complex location of the cylinder center. Defaults to [0.0, 0.0].
    """

    def __init__(self, **kwargs):

        # Set params
        self._R = kwargs.get("cylinder_radius", 1.0)
        z0 = kwargs.get("z0", [0.0, 0.0])
        self._z0 = np.complex(z0[0], z0[1])

        # Call parent initializer
        super().__init__(x_le=np.real(self._z0)-self._R, x_te=np.real(self._z0)+self._R)


    def _geometry(self, x):
        # Returns the camber line and the upper and lower surface coordinates at the x location given

        # Get y values
        dx = x-np.real(self._z0)
        y_upper = np.sqrt(self._R**2-dx**2)+np.imag(self._z0)
        y_lower = -np.sqrt(self._R**2-dx**2)+np.imag(self._z0)
        y_camber = np.zeros_like(x)+np.imag(self._z0)
        
        return np.array([x, y_camber]).T, np.array([x, y_upper]).T, np.array([x, y_lower]).T


    def _velocity(self, point):
        # Returns the velocity components at the Cartesian coordinates given

        # Transform to complex
        z = np.complex(point[0], point[1])
        j = np.complex(0.0, 1.0)

        # Get complex velocity
        w = self._V*(np.exp(-j*self._alpha)+j*self._gamma/(2.0*np.pi*self._V*(z-self._z0))-self._R**2*np.exp(j*self._alpha)/(z-self._z0)**2)

        Vx = np.real(w)
        Vy = -np.imag(w)
        V = np.sqrt(Vx*Vx+Vy*Vy)
        return np.array([Vx, Vy]), 1.0-V*V/self._V**2

    
    def solve(self):
        """Determine the coefficients."""
        return self._gamma/(self._V*self._c)


if __name__=="__main__":

    # Get input
    input_file = sys.argv[-1]
    with open(input_file, 'r') as input_handle:
        input_dict = json.load(input_handle)

    # Initialize object
    geom_dict = input_dict["geometry"]

    # Airfoil
    if "airfoil" in list(geom_dict.keys()):
        airfoil = True
        flow_object = VortexPanelAirfoil(**geom_dict)

        # Print airfoil information
        print("".join(["-"]*80))
        if flow_object._airfoil != "file":
            print("Airfoil: {0}".format(flow_object._airfoil))
            print("    Trailing edge type: {0}".format(geom_dict["trailing_edge"]))
        else:
            print("Airfoil: {0}".format(geom_dict["filename"]))
        print("    # panels: {0}".format(flow_object._N))
        if "UL" in flow_object._airfoil:
            print("    Design lift coefficient: {0}".format(geom_dict["CL_design"]))
        print("".join(["-"]*9))

    # Cylinder
    elif "cylinder_radius" in list(geom_dict.keys()):
        airfoil = False
        flow_object = CylinderInComplexPotentialFlow(**geom_dict)

    # Run commands
    for key, value in input_dict["run_commands"].items():
        
        # Don't run false commands
        if not value:
            continue

        oper_dict = input_dict["operating"]
        plot_dict = input_dict["plot"]

        if key == "alpha_sweep":
            sweep_dict = input_dict["alpha_sweep"]

            # Start table
            print("\nSweeping in alpha...")
            print("{:<20}{:<20}{:<20}{:<20}".format("Alpha [deg]", "CL", "Cm_le", "Cm_c4"))
            print("".join(["-"]*80))

            # Get iteration params
            V = oper_dict["freestream_velocity"]
            a_start = sweep_dict["start[deg]"]
            a_end = sweep_dict["end[deg]"]
            a_inc = sweep_dict["increment[deg]"]
            n = int((a_end-a_start)/a_inc+1)

            # Iterate
            for a in np.linspace(a_start, a_end, n):

                # Set condition
                if airfoil:
                    flow_object.set_condition(alpha=a, V=V)
                else:
                    flow_object.set_condition(**oper_dict)

                # Solve
                coefs = flow_object.solve()
                print("{:<20.12}{:<20.12}{:<20.12}{:<20.12}".format(a, *coefs))

        elif key == "plot_streamlines":

            # Set condition
            print("\nPlotting streamlines...")
            a = oper_dict.get("angle_of_attack[deg]", 0.0)
            if airfoil:
                flow_object.set_condition(alpha=a, V=oper_dict["freestream_velocity"])
            else:
                flow_object.set_condition(**oper_dict)

            # Solve
            coefs = flow_object.solve()
            #print("{:<20.12}{:<20.12}{:<20.12}{:<20.12}".format(a, *coefs))

            # Plot
            flow_object.plot_streamlines(plot_dict["x_start"],
                                         [plot_dict["x_lower_limit"], plot_dict["x_upper_limit"]],
                                         plot_dict["delta_s"],
                                         plot_dict["n_lines"],
                                         plot_dict["delta_y"],
                                         plot_stagnation=True)

        elif key == "plot_pressure":

            # Set condition
            print("\nPlotting pressure...")
            oper_dict = input_dict["operating"]
            a = oper_dict["alpha[deg]"]
            if airfoil:
                flow_object.set_condition(alpha=a, V=oper_dict["freestream_velocity"])
            else:
                flow_object.set_condition(**oper_dict)

            # Solve
            coefs = flow_object.solve()
            #print("{:<20.12}{:<20.12}{:<20.12}{:<20.12}".format(a, *coefs))

            flow_object.plot_C_P()
        
        elif key == "export_geometry":
            print("\nExporting geometry...")
            flow_object.export_geometry()
