import numpy as np

def streamline(start, ds, velocity):
    """Returns an array of points along a streamline in a two-dimensional velocity
    field defined by velocity.

    Parameters
    ----------
    start : list
        x, y coordinates of the starting point for the streamline.

    ds : float
        Step length along the streamline for the integration.

    velocity : callable
        A function returning the velocity components and pressure coefficient at
        each point in the flow field.
    
    Returns
    -------
    ndarray
        A two-dimensional array of points along the streamline.
    """
    pass