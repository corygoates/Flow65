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

        # Format and show plot
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(x_lims)
        plt.ylim(x_lims)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

        # Get stagnation points
        stag_pnt_0, stag_pnt_1 = self._get_stagnation_points()

    
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