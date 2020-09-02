import numpy as np
import matplotlib.pyplot as plt

global x_min
global x_max


def streamline_derivs(point):
    """Returns the derivatives of the streamline with respect to ds at the given point.

    Parameters
    ----------
    point : ndarray
        Coordinates of point.

    Returns
    -------
    ndarray
        Derivative of streamline position with respect to distance.
    """

    # Get velocity
    v,_ = velocity(point)
    V = np.linalg.norm(v)
    return v/V


def streamline(start, ds):
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
        k1 = streamline_derivs(points[-1])
        k2 = streamline_derivs(points[-1]+0.5*ds*k1)
        k3 = streamline_derivs(points[-1]+0.5*ds*k2)
        k4 = streamline_derivs(points[-1]+ds*k3)

        # Integrate
        new = points[-1]+0.166666666666666666666666*(k1+2.0*(k2+k3)+k4)*ds
        points.append(new)

    return np.array(points)


if __name__=="__main__":

    # Define velocity field
    def velocity(point):
        return np.array([1.0, 0.0]), 0.0

    # Define limits
    global x_min
    global x_max
    x_min = -5.0
    x_max = 5.0

    # Plot various streamlines
    starts = [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0], [0.0, 5.0]]
    ds = 0.01
    plt.figure()
    for start in starts:
        points = streamline(start, ds)
        plt.plot(points[:,0], points[:,1], 'b')
        points = streamline(start, -ds)
        plt.plot(points[:,0], points[:,1], 'b')
    plt.show()