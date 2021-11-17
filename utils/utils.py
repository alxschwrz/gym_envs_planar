import numpy as np
from numpy import sin, cos, pi


def point_inside_circle(x, y, center_x, center_y, radius):
    """Check if a point is inside a circle.

    Args:
        x (float): x coordinate of the point.
        y (float): y coordinate of the point.
        center_x (float): x coordinate of the center of the circle.
        center_y (float): y coordinate of the center of the circle.
        radius (float): radius of the circle.

    Returns:
        bool: True if the point is inside the circle.

    """

    dx = abs(x - center_x)
    dy = abs(y - center_y)
    if dx > radius:
        return False
    if dy > radius:
        return False
    if dx + dy <= radius:
        return True
    if dx ** 2 + dy ** 2 <= radius ** 2:
        return True
    else:
        return False
