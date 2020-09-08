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


    def get_streamline(self, start, ds):
        """Returns an array of points along a streamline in a two-dimensional velocity
        field defined by velocity. velocity(point) should be defined externally.

        Parameters
        ----------
        start : ndarray
            x, y coordinates of the starting point for the streamline.

        ds : float
            Step length along the streamline for the integration.

        Returns
        -------
        ndarray
            A two-dimensional array of points along the streamline.
        """
        global x_min
        global x_max

        # Initialize storage
        points = [np.array(start)]

        # Loop
        iterations = 0
        while points[-1][0]>x_min and points[-1][0]<x_max and iterations<1e6:
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


    def _geometry(self, x):
        # Returns the upper and lower surface coordinates at the x location given

        # Check we're within the circle
        if x < self._x_le or x > self._x_te:
            raise RuntimeError("x is outside the circle!")


if __name__=="__main__":

    # Get input
    input_file = sys.argv[-1]
    with open(input_file, 'r') as input_handle:
        input_dict = json.load(input_handle)

    # Initialize object
    geom_dict = input_dict.get("geometry")
    if "cylinder_radius" in list(geom_dict.keys()):
        obj = CylinderInPotentialFlow(radius=geom_dict["cylinder_radius"],
                                      x_le=geom_dict["x_leading_edge"],
                                      x_te=geom_dict["x_trailing_edge"])

    # Set condition
    obj.set_condition(**geom_dict["operating"])