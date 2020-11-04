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

        # Plot geometry in z plane
        y_start = 0.0
        x_space = np.linspace(self._x_le, self._x_te, 1000)
        camber, upper, lower = self._geometry(x_space)
        plt.plot(camber[:,0], camber[:,1], 'k--')
        plt.plot(upper[:,0], upper[:,1], 'k')
        plt.plot(lower[:,0], lower[:,1], 'k')
        plt.plot(np.real(self._zeta_to_z(self._R-self._e)), 0.0, 'kx')
        plt.plot(np.real(self._zeta_to_z(-self._R+self._e)), 0.0, 'kx')

        # Plot in zeta plane
        upper, lower = self._geometry_in_zeta(x_space)
        plt.plot(upper[:,0], upper[:,1], 'r--')
        plt.plot(lower[:,0], lower[:,1], 'r--')
        plt.plot(self._R-self._e, 0.0, 'rx')
        plt.plot(-self._R+self._e, 0.0, 'rx')
        plt.plot(np.real(self._z0), np.imag(self._z0), 'ro')

        # Determine stagnation points
        if plot_stagnation:
            print("Locating stagnation points...", end='', flush=True)
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
        plt.gca().set_aspect('equal', adjustable='box')
        plt.ylim([y_start-dy*(n+1), y_start+dy*(n+1)])
        plt.xlim(x_lims)
        plt.gca().grid(True, which='both')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    
    @abstractmethod
    def _velocity(self, point):
        pass

    @abstractmethod
    def _geometry(self, x):
        pass


class JoukowskiCylinder(ObjectInPotentialFlow):
    """A cylinder in potential flow.

    Parameters
    ----------
    cylinder_radius : float
        Radius of the cylinder.

    zeta_0 : list
        Complex location of the cylinder center. Defaults to [0.0, 0.0].

    epsilon : float
        Eccentricity of the cylinder
    """

    def __init__(self, **kwargs):

        # Set params
        self._R = kwargs.get("cylinder_radius", 1.0)
        z0 = kwargs.get("zeta_0", [0.0, 0.0])
        self._z0 = np.complex(z0[0], z0[1])
        self._e = kwargs.get("epsilon", 0.0)

        super().__init__(x_le=np.real(self._z0)-self._R, x_te=np.real(self._z0)+self._R)

        # Get leading and trailing edge locations
        self._z_te = 2.0*(np.sqrt(self._R**2-np.imag(self._z0)**2)+np.real(self._z0))
        self._z_le = -2.0*(self._R**2-np.imag(self._z0)**2+np.real(self._z0)**2)/(np.sqrt(self._R**2-np.imag(self._z0)**2)-np.real(self._z0))
        self._c = np.real(self._z_te-self._z_le)
        self._z_c4 = self._z_le+0.25*self._c


    def set_condition(self, **kwargs):
        """Sets the operating condition for the airfoil

        Parameters
        ----------
        freestream_velocity : float
            Freestream velocity.

        angle_of_attack[deg] : float
            Angle of attack in degrees.
        """
        
        # Set params
        self._V = kwargs.get("freestream_velocity")
        self._alpha = np.radians(kwargs.get("angle_of_attack[deg]"))
        self._gamma = 4.0*np.pi*self._V*(np.sqrt(self._R**2-np.imag(self._z0)**2)*np.sin(self._alpha)+np.imag(self._z0)*np.cos(self._alpha))


    def _geometry_in_zeta(self, xi):
        theta = np.arccos((xi-np.real(self._z0))/self._R)
        y_u = self._R*np.sin(theta)+np.imag(self._z0)
        y_l = self._R*np.sin(2.0*np.pi-theta)+np.imag(self._z0)
        return np.array([xi, y_u]).T, np.array([xi, y_l]).T


    def _geometry(self, xi):
        # Returns the camber line and the upper and lower surface coordinates at the x location given

        # Determine theta in the zeta plane
        theta = np.arccos((xi-np.real(self._z0))/self._R)
        zeta_u = self._z0+self._R*np.exp(1.0j*(theta))
        zeta_l = self._z0+self._R*np.exp(1.0j*(2.0*np.pi-theta))

        # Determine z locations
        z_u = self._zeta_to_z(zeta_u)
        z_l = self._zeta_to_z(zeta_l)
        z_c = 0.5*(z_u+z_l)

        return np.array([np.real(z_c), np.imag(z_c)]).T, np.array([np.real(z_u), np.imag(z_u)]).T, np.array([np.real(z_l), np.imag(z_l)]).T


    def _velocity(self, point):
        # Returns the velocity components at the Cartesian coordinates given

        # Transform to complex
        z = np.complex(point[0], point[1])
        zeta = self._z_to_zeta(z)

        # Get complex velocity
        w = self._V*(np.exp(-1.0j*self._alpha)+1.0j*self._gamma/(2.0*np.pi*self._V*(zeta-self._z0))-self._R**2*np.exp(1.0j*self._alpha)/(zeta-self._z0)**2)/(1.0-(self._R-self._e)**2/zeta**2)

        Vx = np.real(w)
        Vy = -np.imag(w)
        V = np.sqrt(Vx*Vx+Vy*Vy)
        return np.array([Vx, Vy]), 1.0-V*V/self._V**2


    def _zeta_to_z(self, zeta):
        # Transforms a point from the zeta plane to the z plane
        return zeta+(self._R-self._e)**2/zeta


    def _z_to_zeta(self, z):
        # Transforms a point from the z plane to the zeta plane
        z1 = z**2-4.0*(self._R-self._e)**2

        zeta1 = np.where(np.real(z1)>0.0, 0.5*(z+np.sqrt(z1)),
                         np.where(np.real(z1)<0.0, 0.5*(z-1.0j*np.sqrt(-z1)),
                                  np.where(np.imag(z1)>=0.0, 0.5*(z+np.sqrt(-z1)), 0.5*(z-1.0j*np.sqrt(-z1)))))

        zeta2 = np.where(np.real(z1)>0.0, 0.5*(z-np.sqrt(z1)),
                         np.where(np.real(z1)<0.0, 0.5*(z+1.0j*np.sqrt(-z1)),
                                  np.where(np.imag(z1)>=0.0, 0.5*(z-np.sqrt(-z1)), 0.5*(z+1.0j*np.sqrt(-z1)))))

        return np.where(np.abs(zeta2-self._z0)>np.abs(zeta1-self._z0), zeta2, zeta1)


    def solve(self):
        """Solves for the airfoil coefficients at the current condition.

        Returns
        -------
        CL

        Cm0

        Cm_c4
        """

        # Lift coefficient
        CL = 2.0*self._gamma/(self._V*self._c)

        # Moment coefficient about origin
        x0 = np.real(self._z0)
        y0 = np.imag(self._z0)
        R = self._R
        Cm0 = 0.25*np.pi*((R**2-y0**2-x0**2)/(R**2-y0**2))**2*np.sin(2.0*self._alpha)
        Cm0 -= 0.25*CL*(x0*np.cos(self._alpha)+y0*np.sin(self._alpha))/(R**2-y0**2)*(np.sqrt(R**2-y0**2)-x0)

        # Moment coefficient about quarter-chord
        x_c4 = np.real(self._z_c4)
        y_c4 = np.imag(self._z_c4)
        Cm_c4 = 0.25*np.pi*((R**2-y0**2-x0**2)/(R**2-y0**2))**2*np.sin(2.0*self._alpha)
        Cm_c4 += 0.25*CL*((x_c4-x0)*np.cos(self._alpha)+(y_c4-y0)*np.sin(self._alpha))/(R**2-y0**2)*(np.sqrt(R**2-y0**2)-x0)

        return CL, Cm0, Cm_c4


    def export_geometry(self, N, filename):
        """Exports the discrete geometry of the airfoil.

        Parameters
        ---------
        N : int
            Number of points.

        filename : str
            Filename to write geometry to.
        """

        # Generate even number of points
        if N%2 == 0:
            d_theta = np.pi/((N/2)-0.5)
            theta = np.linspace(0.5*d_theta, np.pi, int(N/2))
            x = self._x_le+0.5*(1.0-np.cos(theta))*(self._x_te-self._x_le)

        # Generate odd number of points
        else:
            theta = np.linspace(0.0, np.pi, int(N/2)+1)
            x = self._x_le+0.5*(1.0-np.cos(theta))*(self._x_te-self._x_le)

        # Get raw outline points
        _, p_upper, p_lower = self._geometry(x)

        # Initialize point array
        points = np.zeros((N, 2))

        # Organize nodes
        if N%2 == 0:
            points[:int(N/2),:] = p_lower[::-1,:]
            points[int(N/2):,:] = p_upper

            points = points-0.5*(points[int(N/2)-1]+points[int(N/2)])
        else:
            points[:int(N/2)+1,:] = p_lower[::-1,:]
            points[int(N/2)+1:,:] = p_upper[1:,:]

            points = points-points[int(N/2)]

        # Scale to unit chord
        points = points/self._c

        # Export points
        header = "{:<30} {:<30}".format('x', 'y')
        fmt_str = "%30.25e %30.25e"
        np.savetxt(filename, points, fmt=fmt_str, header=header)


class JoukowskiAirfoil(JoukowskiCylinder):
    """Uses the Joukowski transformation to create an airfoil.

    Parameters
    ----------
    design_CL : float
        Design lift coefficient at 0 degrees angle of attack.

    design_thickness : float
        Design thickness as a percentage of the chord.
    
    cylinder_radius : float
        Cylinder radius.
    """

    def __init__(self, **kwargs):

        # Get kwargs
        CL_d = kwargs["design_CL"]
        t_d = kwargs["design_thickness"]
        R = kwargs["cylinder_radius"]

        # Determine complex offset
        xi_0 = -4.0*R*t_d/(3.0*np.sqrt(3.0))
        eta_0 = CL_d*R/(2.0*np.pi*(1.0-xi_0/R))

        # Determine eccentricity
        eps = R-np.sqrt(R**2-eta_0**2)-xi_0

        # Initialize
        super().__init__(cylinder_radius=R, zeta_0=[xi_0, eta_0], epsilon=eps)

    
if __name__=="__main__":

    # Get input
    input_file = sys.argv[-1]
    with open(input_file, 'r') as input_handle:
        input_dict = json.load(input_handle)

    # get params
    geom_dict = input_dict["geometry"]
    airfoil = geom_dict["type"] == "airfoil"

    # Initialize object
    if airfoil:

        flow_object = JoukowskiAirfoil(**geom_dict)

        # Check leading and trailing edge locations
        print("z_LE: {0}".format(flow_object._z_le))
        print("z_TE: {0}".format(flow_object._z_te))
        print("z_c4: {0}".format(flow_object._z_c4))

    else:
        flow_object = JoukowskiCylinder(**geom_dict)

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
                flow_object.set_condition(**oper_dict)

                # Solve
                coefs = flow_object.solve()
                print("{:<20.12}{:<20.12}{:<20.12}{:<20.12}".format(a, *coefs))

        elif key == "plot_streamlines":

            # Set condition
            print("\nPlotting streamlines...")
            a = oper_dict.get("angle_of_attack[deg]", 0.0)
            flow_object.set_condition(**oper_dict)

            # Plot
            flow_object.plot_streamlines(plot_dict["x_start"],
                                         [plot_dict["x_lower_limit"], plot_dict["x_upper_limit"]],
                                         plot_dict["delta_s"],
                                         plot_dict["n_lines"],
                                         plot_dict["delta_y"],
                                         plot_stagnation=True)
            coefs = flow_object.solve()
            print("CL: {0}".format(coefs[0]))
            print("Cm0: {0}".format(coefs[1]))
            print("Cm_c4: {0}".format(coefs[2]))

        elif key == "plot_pressure":

            # Set condition
            print("\nPlotting pressure...")
            oper_dict = input_dict["operating"]
            a = oper_dict["alpha[deg]"]
            flow_object.set_condition(**oper_dict)

            flow_object.plot_C_P()
        
        elif key == "export_geometry":
            print("\nExporting geometry...")
            flow_object.export_geometry(geom_dict["output_points"], value)