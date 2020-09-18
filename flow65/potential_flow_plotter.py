import sys
import json

import numpy as np
import matplotlib.pyplot as plt

from abc import abstractmethod


class PotentialFlowElement:
    """A base class for potential flow elements."""

    def velocity(self, point):
        """Returns the velocity vector at the given point due to this element."""

        # Get r and theta
        dx = point[0]-self._x
        dy = point[1]-self._y
        r = np.sqrt(dx**2+dy**2)
        theta = np.arctan2(dy, dx)

        # Get cylindrical components
        V_r, V_theta = self._vel_in_cyl(r, theta)

        # Get Cartesian components
        C_theta = np.cos(theta)
        S_theta = np.sin(theta)
        return np.array([V_r*C_theta-V_theta*S_theta, V_r*S_theta+V_theta*C_theta])


    @abstractmethod
    def _vel_in_cyl(self, r, theta):
        # Returns the velocity components in cylindrical coordinates
        pass


class Freestream(PotentialFlowElement):
    """A freestream potential flow.

    Parameters
    ----------
    velocity : float

    angle_of_attack : float
        Given in degrees.
    """

    def __init__(self, **kwargs):
        self._V = kwargs["velocity"]
        self._a = np.radians(kwargs["angle_of_attack"])
        self._x = 0.0
        self._y = 0.0


    def _vel_in_cyl(self, r, theta):
        return self._V*np.cos(theta-self._a), -self._V*np.sin(theta-self._a)

    def plot(self):
        pass


class Doublet(PotentialFlowElement):
    """A potential doublet.

    Parameters
    ----------
    x : float

    y : float

    kappa : float

    alpha : float
        Given in degrees.
    """

    def __init__(self, **kwargs):
        self._x = kwargs["x"]
        self._y = kwargs["y"]
        self._k = kwargs["kappa"]
        self._a = np.radians(kwargs["alpha"])


    def _vel_in_cyl(self, r, theta):
        return -self._k/(2.0*np.pi)*np.cos(theta-self._a)/r**2, -self._k/(2.0*np.pi)*np.sin(theta-self._a)/r**2

    
    def plot(self):
        plt.plot(self._x, self._y, "Dr")


class Source(PotentialFlowElement):
    """A potential source.

    Parameters
    ----------
    x : float

    y : float

    lambda : float
    """

    def __init__(self, **kwargs):
        self._x = kwargs["x"]
        self._y = kwargs["y"]
        self._l = kwargs["lambda"]


    def _vel_in_cyl(self, r, theta):
        return self._l/(2.0*np.pi*r), 0.0


    def plot(self):
        if self._l >= 0:
            plt.plot(self._x, self._y, "ob")
        else:
            plt.plot(self._x, self._y, "or")


class Vortex(PotentialFlowElement):
    """A potential vortex.

    Parameters
    ----------
    x : float

    y : float

    gamma : float
    """

    def __init__(self, **kwargs):
        self._x = kwargs["x"]
        self._y = kwargs["y"]
        self._g = kwargs["gamma"]


    def _vel_in_cyl(self, r, theta):
        return 0.0, -self._g/(2.0*np.pi*r)


    def plot(self):
        if self._g >= 0:
            plt.plot(self._x, self._y, "xb")
        else:
            plt.plot(self._x, self._y, "xr")


class PotentialFlowField:
    """A 2D potential flow field.

    Parameters
    ----------
    elementss : dict
        A dictionary of potential flow elements.
    """

    def __init__(self, **kwargs):
        
        # Get elements
        self._element_inputs = kwargs["elements"]
        self._elements = []
        for _, value in self._element_inputs.items():
            element_type = value["type"]
            if element_type == "freestream":
                self._elements.append(Freestream(**value))
            elif element_type == "source":
                self._elements.append(Source(**value))
            elif element_type == "doublet":
                self._elements.append(Doublet(**value))
            elif element_type == "vortex":
                self._elements.append(Vortex(**value))


    def _get_streamline_derivs(self, point):
        # Returns the derivatives of the streamline with respect to ds at the given point.

        # Get velocity
        v = np.zeros(2)
        for element in self._elements:
            v += element.velocity(point)
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

        # Plot other streamlines
        for y in np.linspace(-dy*n, dy*n, 2*n+1):
            point = np.array([x_start, y])
            streamline = self.get_streamline(point, 0.01, x_lims)
            plt.plot(streamline[:,0], streamline[:,1], 'k-')
            streamline = self.get_streamline(point, -0.01, x_lims)
            plt.plot(streamline[:,0], streamline[:,1], 'k-')

        # Plot signularities
        for element in self._elements:
            element.plot()

        # Format and show plot
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(x_lims)
        plt.ylim([-dy*(n+1), dy*(n+1)])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


if __name__=="__main__":

    # Get input
    input_file = sys.argv[-1]
    with open(input_file, 'r') as input_handle:
        input_dict = json.load(input_handle)

    # Initialize field
    field = PotentialFlowField(elements=input_dict["elements"])

    # Plot
    plot_dict = input_dict["plot"]
    field.plot(plot_dict["x_start"],
               [plot_dict["x_lower_limit"], plot_dict["x_upper_limit"]],
               plot_dict["delta_s"],
               plot_dict["n_lines"],
               plot_dict["delta_y"])