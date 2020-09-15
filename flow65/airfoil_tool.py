import sys
import json

import numpy as np
import matplotlib.pyplot as plt

from abc import abstractmethod


class ObjectInPotentialFlow:
    """An object in a 2D potential flow.

    Parameters
    ----------
    radius : float
        Radius of the cylinder.

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
        max_iterations = int(16*(x_lims[1]-x_lims[0])/ds)

        # Initialize storage
        points = [np.array(start)]

        # Loop
        iterations = 0
        while points[-1][0]>x_lims[0] and points[-1][0]<x_lims[1] and iterations<max_iterations:
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

        # Get velocity
        V_u = self._velocity(p_u)
        V_l = self._velocity(p_l)

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
        for x in x_space:
            V_T_u, V_T_l = self._surface_tangential_velocity(x)
            plt.plot(x, V_T_l, 'b.')
            plt.plot(x, V_T_u, 'r.')

        # Format and show plot
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(x_lims)
        plt.ylim(x_lims)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

        self._stagnation()

    
    @abstractmethod
    def _velocity(self, point):
        pass

    @abstractmethod
    def _geometry(self, x):
        pass


class CylinderInPotentialFlow(ObjectInPotentialFlow):
    """A cylinder in potential flow.

    Parameters
    ----------
    radius : float
        Radius of the cylinder.

    x_le : float
        x coordinate of the leading edge.

    x_te : float
        x coordinate of the trailing edge.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set params
        self._R = kwargs.get("radius", 1.0)
        self._x_cent = 0.5*(self._x_te+self._x_le)

        # Check radius
        R = 0.5*(self._x_te-self._x_le)
        if abs(self._R-R) > 1e-10:
            raise IOError("The radius of the circle does not match up with the distance between the leading- and trailing-edge locations.")


    def _geometry(self, x):
        # Returns the camber line and the upper and lower surface coordinates at the x location given

        # Get y values
        dx = x-self._x_cent
        y_upper = np.sqrt(self._R**2-dx**2)
        y_lower = -np.sqrt(self._R**2-dx**2)
        y_camber = np.zeros_like(x)
        
        return np.array([x, y_camber]).T, np.array([x, y_upper]).T, np.array([x, y_lower]).T


    def _velocity(self, point):
        # Returns the velocity components at the Cartesian coordinates given

        # Get cylindrical coordinates
        r = np.sqrt(point[0]**2+point[1]**2)
        theta = np.arctan2(point[1], point[0])

        # Get cylindrical components
        R_r = (self._R/r)**2
        V_r = self._V*(1.0-R_r)*np.cos(theta-self._alpha)
        V_theta = -self._V*(1.0+R_r)*np.sin(theta-self._alpha)

        # Get Cartesian components
        C_theta = np.cos(theta)
        S_theta = np.sin(theta)
        return np.array([V_r*C_theta-V_theta*S_theta, V_r*S_theta+V_theta*C_theta])


if __name__=="__main__":

    # Get input
    input_file = sys.argv[-1]
    with open(input_file, 'r') as input_handle:
        input_dict = json.load(input_handle)

    # Initialize object
    geom_dict = input_dict["geometry"]
    if "cylinder_radius" in list(geom_dict.keys()):
        obj = CylinderInPotentialFlow(radius=geom_dict["cylinder_radius"],
                                      x_le=geom_dict["x_leading_edge"],
                                      x_te=geom_dict["x_trailing_edge"])

    # Set condition
    obj.set_condition(**input_dict["operating"])

    # Plot
    plot_dict = input_dict["plot"]
    obj.plot(plot_dict["x_start"],
             [plot_dict["x_lower_limit"], plot_dict["x_upper_limit"]],
             plot_dict["delta_s"],
             plot_dict["n_lines"],
             plot_dict["delta_y"])