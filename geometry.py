#!/usr/bin/env python
"""
Functions for working with geometry.
"""

import numpy as np
import constants as const
import matplotlib._cntr as cntr
import matplotlib.path as mplPath


def cart2pol(*args):
    """Convert cartesian to polar coordinates.

    If one arg is passed, assumed to be N_points x 2 array with x in the first
    column and y in the second column.

    If two args are passed, first is assumed to be x and second y.

    Returns
    -------
    r, theta

    """
    if len(args) == 1:
        x = args[:, 0]
        y = args[:, 1]
    else:
        x = args[0]
        y = args[1]
    return np.hypot(x, y), np.arctan2(y, x)


def sph2cart(r, theta, phi):
    """Convert spherical to cartesian coordinates.

    Parameters
    ----------
    r
        distance from origin
    theta
        zenith/co-latitude angle (out/down from the +z-axis)
    phi
        azimuth/longitude angle (positive is counter-clockwise about equator)

    Returns
    -------
    x, y, z
        Cartesian coordinate(s)

    """
    sintheta = np.sin(theta)
    return r*sintheta*np.cos(phi), r*sintheta*np.sin(phi), r*np.cos(theta)


def pol2cart(*args):
    """Convert plane polar to cartesian coordinates.

    If only one argument is supplied, it must be a N_r x 2 array where `r` is
    in the first column and `theta` is in the second colum.

    Parameters
    ----------
    array of [r[:], theta[:]] or r, theta passed separately

    Returns
    -------
    x, y

    """
    if len(args) == 1:
        r = args[0][:, 0]
        theta = args[0][:, 1]
    else:
        assert len(args) == 2
        r = args[0]
        theta = args[1]
    return r*np.cos(theta), r*np.sin(theta)


def length(x, y, z):
    return np.sqrt(x*x + y*y + z*z)


def angle_between_cart(x1, y1, z1, x2, y2, z2):
    len1 = length(x1, y1, z1)
    len2 = length(x2, y2, z2)
    return np.arccos((x1*x2 + y1*y2 + z1*z2) / (len1 * len2))


def angle_between_sph(theta1, phi1, theta2, phi2):
    x1, y1, z1 = sph2cart(1, theta1, phi1)
    x2, y2, z2 = sph2cart(1, theta2, phi2)
    return angle_between_cart(x1, y1, z1, x2, y2, z2)


def polyArea(x, y):
    """Vertices must be an N_vert x 2 Numpy array; first vertex need not be
    the same as the last index (i.e., the algorithm "closes" the polygon for
    you).

    From
    http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    user Mahdi

    """
    return 0.5*np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


#def polyArea(vert):
#    """Vertices must be an N_vert x 2 Numpy array; first vertex need not be
#    the same as the last index (i.e., the algorithm "closes" the polygon for
#    you).
#
#    See http://paulbourke.net/geometry/polygonmesh for reference
#
#    """
#    vert1 = np.roll(vert, shift=-1, axis=0)
#    xi = vert[:, 0]
#    yi = vert[:, 1]
#    xip1 = vert1[:, 0]
#    yip1 = vert1[:, 1]
#
#    return 0.5 * np.sum(xi*yip1 - xip1*yi)


def polyCentroid(vert):
    """Vertices must be an N_vert x 2 Numpy array; first vertex need not be
    the same as the last index (i.e., the algorithm "closes" the polygon for
    you).

    See http://paulbourke.net/geometry/polygonmesh for reference"""
    if isinstance(vert, mplPath.Path):
        vert = vert.vertices

    vert1 = np.roll(vert, shift=-1, axis=0)
    xi = vert[:, 0]
    yi = vert[:, 1]
    xip1 = vert1[:, 0]
    yip1 = vert1[:, 1]
    area = 0.5 * np.sum( xi*yip1 - xip1*yi )
    cx = 1/(6*area) * np.sum( (xi+xip1)*(xi*yip1-xip1*yi) )
    cy = 1/(6*area) * np.sum( (yi+yip1)*(xi*yip1-xip1*yi) )

    return np.array([cx, cy])


def expandPolyApprox(vert, perp_dist):
    """Approximate expansion of polygon by a perpendicular distance on all
    sides. A negative value for perp_dist contracts the polygon.

    """
    centroid = polyCentroid(vert)

    r0, theta = cart2pol(vert[:, 0] - centroid[0], vert[:, 1] - centroid[1])

    theta_ = theta.copy()
    theta_.sort()

    # Find average angle (rays from centroid) separating vertices
    dtheta = np.mean((theta_ - np.roll(theta_, shift=1, axis=0)) % (2*np.pi))

    # Half that angle is the angle to the center of each edge from the
    # centroid, taken (erroneously, but good enough for now) to be the
    # perpendicular. Obviously a worse approximation the less regular the
    # polygon.
    rprime = r0 + perp_dist/np.cos(dtheta/2.0)

    xprime, yprime = pol2cart(rprime, theta)

    return np.array([xprime+centroid[0], yprime+centroid[1]]).T


def contractPolyApprox(vert, perp_dist):
    """Convenience function. See expandPolyApprox for details; this just
    does the opposite, feeding -perp_dist to that function."""
    return expandPolyApprox(vert, -perp_dist)


def cherenkovAngle(n, speed, light_vac_speed=const.c):
    """Returns the Cherenkov angle, in radians, for a particle moving in a
    medium with index of refraction n at the given speed."""
    ## TODO: add warning if the following holds! (but it's here because someone
    ## might say speed = 3e8 instead of specifying const.c
    if speed > light_vac_speed:
        speed = light_vac_speed
    beta = speed / light_vac_speed
    cos_theta = 1/(n*beta)
    return np.arccos(cos_theta)


def contourPaths(x, y, z, z_thresh):
    """Returns list of matplotlib path(s) representing contour line(s) at
    z_thresh. Credit:Ian Thomas-8
    http://matplotlib.1069221.n5.nabble.com/pyplot-Extract-contourset-without-plotting-td15868.html
    """
    contour_vertices = cntr.Cntr(x, y, z)
    nlist = contour_vertices.trace(z_thresh, z_thresh, 0)
    segments = nlist[:len(nlist)//2]
    paths = [mplPath.Path(s) for s in segments]
    return paths


def main():
    """Test some stuff"""
    import matplotlib.pyplot as plt

    vert = np.array(
      [[-10.97,   6.72],
       [-22.23, -29.97],
       [-14.31, -67.24],
       [ 15.13, -99.93],
       [ 36.65, -95.36],
       [ 79.69, -86.21],
       [101.21, -81.63],
       [114.81, -39.79],
       [106.89,  -2.52],
       [ 77.45,  30.17],
       [ 55.93,  25.6 ],
       [ 12.89,  16.45],
       [-10.97,   6.72]]
    )

    cntr = polyCentroid(vert)
    print cntr

    fig = plt.figure(200)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.add_patch(plt.Polygon(vert, closed=True, fill=True,
                             facecolor='w', edgecolor='k'))
    ax.plot(cntr[0], cntr[1], 'ko')
    plt.draw()
    plt.show()

    vert = np.array(
        [[-1, -1],
         [ 1, -1],
         [ 1,  1],
         [-1,  1]]
    )

    cntr = polyCentroid(vert)
    print 'square centered at (0, 0): computed centroid =', cntr

    fig = plt.figure(201)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.add_patch(
        plt.Polygon(vert, closed=True, fill=True, facecolor='w', edgecolor='k')
    )
    ax.plot(cntr[0], cntr[1], 'ko')
    plt.draw()
    plt.show()


if __name__ == "__main__":
    main()
