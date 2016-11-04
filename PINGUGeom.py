#!/usr/bin/env python

import os, sys
import numpy as np
import pandas as pd
try:
    from scipy.spatial import ConvexHull
except:
    pass
import matplotlib as mpl

import geometry
import constants as const

#import matplotlib.nxutils as nx

class PINGUGeom:
    def __init__(self, version, include_dc, innerPad=0, outerPad=0):
        self.CONT_Z_RANGE = {
                26: [-500, -180],
                22: [-500, -180],
                24: [-500, -180],
                30: [-500, -180],
                27: [-500, -180],
                15: [-500, -180],
                17: [-500, -180],
                29: [-500, -180],
                23: [-500, -180],
                25: [-500, -180],
                31: [-500, -180],
                28: [-500, -180],
                36: [-500, -180],
                38: [-500, -180],
                39: [-500, -180],
        }[version]
        self.CONT_XY_CENTER = {
                26: [45, -38],
                22: [50, -35],
                24: [50, -35],
                30: [50, -35],
                27: [44, -40],
                15: [50, -50],
                17: [50, -50],
                29: [50, -35],
                23: [50, -50],
                25: [50, -50],
                31: [50, -50],
                28: [50, -45],
                36: [50, -35],
                38: [50, -35],
                39: [50, -35],
        }[version]
        self.CONT_XY_RADIUS = {
                26: 25,
                22: 55,
                24: 55,
                30: 55,
                27: 60,
                15: 75,
                17: 75,
                29: 90,
                23: 90,
                25: 90,
                31: 90,
                28: 175,
                36: 100,
                38: 100,
                39: 100,
        }[version]
        all_xy_radius_step2 = {
                36: 85,
                38: 85,
                39: 85,
        }
        self.CONT_STEP2_RAD = all_xy_radius_step2[version] if version in all_xy_radius_step2 else None

        # TODO: Version 15 geometry is not satisfactorily defined!!!
        if version == 15:
            #-- Regular hexagonal bounding box for main part of PINGU
            #   NOTE the actual geom is *NOT* a regular hexagon, and is
            #   computed by loading the actual string locations from the
            #   below-named file
            self.ROT     =    0.0*np.pi/180
            self.RHSHIFT =   -4.0
            self.RVSHIFT =    0.0
            #self.X0      =   50.0 + self.RHSHIFT*np.cos(self.ROT) - self.RVSHIFT*np.sin(self.ROT)
            #self.Y0      =  -34.0 + self.RHSHIFT*np.sin(self.ROT) + self.RVSHIFT*np.cos(self.ROT)
            self.X0      =    46.40885489
            self.Y0      =   -35.02185522
            self.RMAX    =   78.0
            self.RMIN    =   self.RMAX * np.cos(np.pi/6)
            self.ZMIN    = -466.0
            self.ZMAX    = -181.0
            self.ZMID    = (self.ZMIN+self.ZMAX)/2.0
            self.GEOMFILE = os.path.join(os.path.expanduser('~'), 'cowen/scripts/sd_xyz_PINGU_v15.csv')
            #-- Note that this is 0-indexed, whereas IceCube string numbers are
            #   1-indexed
            self.N_STRINGS = 40
            self.N_DOM_PER_STRING = 60
            self.ICECUBE_STRINGIND = np.arange(1, 78+1, dtype=int)
            self.DEEPCORE_STRINGIND = np.arange(79, 86+1, dtype=int)
            self.PINGU_STRINGIND = np.arange(87, 87+self.N_STRINGS, dtype=int)

        elif version in [36, 38, 39]:
            #-- Regular hexagonal bounding box for main part of PINGU
            #   NOTE the actual geom is *NOT* a regular hexagon, and is
            #   computed by loading the actual string locations from the
            #   below-named file
            self.ROT     =   12.0*np.pi/180
            self.RHSHIFT =   -4.0
            self.RVSHIFT =    0.0
            #self.X0      =   50.0 + self.RHSHIFT*np.cos(self.ROT) - self.RVSHIFT*np.sin(self.ROT)
            #self.Y0      =  -34.0 + self.RHSHIFT*np.sin(self.ROT) + self.RVSHIFT*np.cos(self.ROT)
            self.X0      =   46.40885489
            self.Y0      =  -35.02185522
            self.CUT_X0  =   50
            self.CUT_Y0  =  -35
            self.RMAX    =   78.0
            self.RMIN    =   self.RMAX * np.cos(np.pi/6)
            self.ZMIN    = -466.0
            self.ZMAX    = -181.0
            self.CUT_ZMIN= -500
            self.CUT_ZMAX= -180
            self.ZMID    = (self.ZMIN+self.ZMAX)/2.0
            self.GEOMFILE = os.path.join(
                os.path.expanduser('~'),
                'cowen/scripts/sd_xyz_PINGU_v%d.csv' % version
            )
            #-- Note that this is 0-indexed, whereas IceCube string numbers are
            #   1-indexed
            self.N_STRINGS = 40
            self.N_DOM_PER_STRING = 96
            self.N_PINGU_DOMS = self.N_STRINGS * self.N_DOM_PER_STRING
            self.ICECUBE_STRINGIND = np.arange(1, 78+1, dtype=int)
            self.DEEPCORE_STRINGIND = np.arange(79, 86+1, dtype=int)
            self.PINGU_STRINGIND = np.arange(87, 87+self.N_STRINGS, dtype=int)

        else:
            raise Exception('Version ' + str(version) + ' geometry unknown.')

        self.loadDOMXYZ()
        self.computeXYHull(include_dc=include_dc)

        self.innerPad = innerPad
        self.outerPad = outerPad

        if innerPad == 0:
            self.innerHullVert = self.hullXY
            self.innerHullPath = self.hullPath
        else:
            self.innerHullVert = geometry.contractPolyApprox(self.hullXY, innerPad)
            self.innerHullPath = mpl.path.Path(self.innerHullVert)

        if outerPad == 0:
            self.outerHullVert = self.hullXY
            self.outerHullPath = self.hullPath
        else:
            self.outerHullVert = geometry.expandPolyApprox(self.hullXY, outerPad)
            self.outerHullPath = mpl.path.Path(self.outerHullVert)
        self.innerHullArea = geometry.polyArea(self.innerHullVert[:,0], self.innerHullVert[:,1])
        self.outerHullArea = geometry.polyArea(self.outerHullVert[:,0], self.outerHullVert[:,1])
        self.innerHullVol = self.innerHullArea * (self.ZMAX - self.ZMIN)
        self.outerHullVol = self.outerHullArea * (self.ZMAX - self.ZMIN)

    @staticmethod
    def uniqueDOMID(string_num, dom_num):
        return np.int_(string_num*10000 + dom_num)

    def xyDistFromCentroid(self, x, y):
        dx = x-self.centroidXY[0]
        dy = y-self.centroidXY[1]
        return np.sqrt((dx)**2 + (dy)**2)

    def zDistFromCentroid(self, z):
        return z - (self.ZMAX + zelf.ZMIN)/2.0

    def xyDistFromHullEdge(self, x, y):
        """Sign convention: Positive is inside, and negative is outside"""
        singleton = False
        if not hasattr(x, '__iter__'):
            x = [x]
            y = [y]
            singleton = True
        circVertXY = np.append(self.hullXY[:,:],[self.hullXY[0,:]],axis=0)
        edgeDeltaXY = np.diff(circVertXY, axis=0)
        edMagSq = edgeDeltaXY[:,0]**2 + edgeDeltaXY[:,1]**2
        #edMag = np.sqrt(edMagSq)
        minDist = []
        for (x,y) in zip(x,y):
            pointVec = np.ones(np.shape(self.hullXY)) * [x,y]
            firstVertToPoint = pointVec - circVertXY[0:self.nHullVert,:]
            secondVertToPoint = pointVec - circVertXY[1:,:]
            fvtpMagSq = firstVertToPoint[:,0]**2 + firstVertToPoint[:,1]**2
            svtpMagSq = secondVertToPoint[:,0]**2 + secondVertToPoint[:,1]**2
            dotProd = edgeDeltaXY[:,0] * firstVertToPoint[:,0] + edgeDeltaXY[:,1] * firstVertToPoint[:,1]
            #normedDotProd = dotProd / fvtpMagSq
            normedDotProd = dotProd / edMagSq
            #print 'r (normedDotProd)', normedDotProd
            #print 'p0-p1', fvtpMagSq
            #print 'p0-p2', svtpMagSq
            distances = (normedDotProd <= 0) * np.sqrt(fvtpMagSq) + \
                    (normedDotProd >= 1)*np.sqrt(svtpMagSq) + \
                    ((normedDotProd > 0) & (normedDotProd < 1))*np.sqrt(np.abs(fvtpMagSq-normedDotProd**2*edMagSq))
            #print distances
            isInside = self.insideHullXY(x,y)[0]
            minDist.append(np.min(distances)*(isInside*1+np.logical_not(isInside)*-1))
        if singleton:
            minDist = minDist[0]
        return minDist

    def zDistFromHullEdge(self, z):
        """Sign convention: Positive is inside, and negative is outside"""
        singleton = False
        if not hasattr(z, '__iter__'):
            z = [z]
            singleton = True
        minDist = []
        for z in z:
            if z >= self.ZMID:
                minDist.append(self.ZMAX-z)
            else:
                minDist.append(z-self.ZMIN)
        if singleton:
            minDist = minDist[0]
        return minDist

    def xyDistToNearestDOM(self, x, y):
        singleton = False
        if not hasattr(x, '__iter__'):
            x = [x]
            y = [y]
            singleton = True
        minDist = []
        for (x,y) in zip(x,y):
            minDist.append(np.min(np.sqrt((x-self.XY[:,0])**2 + (y-self.XY[:,1])**2)))
        if singleton:
            minDist = minDist[0]
        return minDist

    def xyComponentsToNearestDOM(self, x, y):
        singleton = False
        if not hasattr(x, '__iter__'):
            x = [x]
            y = [y]
            singleton = True
        closestXYComps = []
        for (x,y) in zip(x,y):
            xComp = x-self.XY[:,0]
            yComp = y-self.XY[:,1]
            dist = np.sqrt(xComp**2 + yComp**2)
            closestInd = np.where(dist == np.min(dist))[0][0]
            closestXYComps.append((xComp[closestInd], yComp[closestInd]))
        if singleton:
            closestXYComps = closestXYComps[0]
        return closestXYComps

    def zComponentFromNearestDOM(self, z):
        singleton = False
        if not hasattr(z, '__iter__'):
            z = [z]
            singleton = True
        shortestZComp = []
        for z in z:
            zcomp = z-self.Z
            zdist = np.abs(zcomp)
            minInd = np.where(zdist == min(zdist))[0]
            shortestZComp.append(zcomp[minInd][0])
        if singleton:
            shortestZComp = shortestZComp[0]
        return shortestZComp

    def distToNearestDOM(self, x, y, z):
        singleton = False
        if not hasattr(x, '__iter__'):
            x = [x]
            y = [y]
            z = [z]
            singleton = True
        xyComp = np.array(self.xyComponentsToNearestDOM(x, y))
        zComp = np.array(self.zComponentFromNearestDOM(z))
        minDist = np.sqrt((xyComp**2).sum(axis=1) + zComp**2)
        if singleton:
            minDist = minDist[0]
        return minDist

    def sphCoordsFromNearestDOM(self, x, y, z):
        singleton = False
        if not hasattr(x, '__iter__'):
            x = np.array([x])
            y = np.array([y])
            z = np.array([z])
            singleton = True
        xyComponents = np.array(self.xyComponentsToNearestDOM(x, y))
        zComponent = np.array(self.zComponentFromNearestDOM(z))
        rho2 = (xyComponents[:,0]**2 + xyComponents[:,1]**2) # projected length in the plane
        #print 'rho2 shape:', np.shape(rho2), 'zComponent shape:', np.shape(zComponent), 'x shape:', np.shape(x), 'xyComponents shape:', np.shape(xyComponents)
        r = np.sqrt(rho2 + zComponent**2)    # distance from DOM
        # TODO: this defines the co-latitutde (standard theta angle used for
        # physics coordinate systems, but **NOT** the latitude!
        theta = np.arctan2(np.sqrt(rho2), zComponent) # **CO-**lat (0 above, pi below dom)
        phi = np.arctan2(xyComponents[:,1], xyComponents[:,0]) # longitude
        if singleton:
            r = r[0]
            theta = theta[0]
            phi = phi[0]
        return r, theta, phi

    def distFromHullEdge(self, x, y, z):
        """Sign convention: Positive is inside, and negative is outside"""
        xyDist = self.xyDistFromHullEdge

    def insideZ(self, z, innerPad=None):
        if innerPad is None:
            innerPad = self.innerPad

        zmax = self.ZMAX - innerPad
        zmin = self.ZMIN + innerPad

        return (z < zmax) & (z > zmin)

    def outsideZ(self, z, outerPad=None):
        if outerPad is None:
            outerPad = self.outerPad

        zmax = self.ZMAX + outerPad
        zmin = self.ZMIN - outerPad

        return (z >= zmax) | (z <= zmin)

    #-- TODO : Make all functions take Nx2 or Nx3 arrays for x & leave off
    #   either (y) or (y & z), as is done for the hull methods
    def insideHexXY(self, x, y, innerPad=None):
        if innerPad is None:
            innerPad = self.innerPad

        rmin = self.RMAX * np.cos(np.pi/6) - innerPad
        dx = x-self.X0
        dy = y-self.Y0
        r = np.sqrt((dx)**2 + (dy)**2)
        theta = np.arctan2(dy,dx)
        alpha = ((theta-self.ROT) % (np.pi/3)) - np.pi/6

        return r < rmin/np.cos(alpha)

    def insideHex(self, x, y, z, innerPad=None):
        return self.insideHexXY(x, y, innerPad=innerPad) & \
                self.insideZ(z, innerPad=innerPad)

    def outsideHexXY(self, x, y, outerPad=None):
        if outerPad is None:
            outerPad = self.outerPad

        return np.logical_not(self.insideHexXY(x, y, innerPad=-outerPad))

    def outsideHex(self, x, y, z, outerPad=None):
        return self.outsideHexXY(x, y, outerPad=outerPad) | \
                self.outsideZ(z, outerPad=outerPad)

    def insideHullXY(self, x, y=None, innerPad=None):
        if innerPad is None:
            path = self.innerHullPath
        elif innerPad == 0:
            path = self.hullPath
        else:
            verts = geometry.contractPolyApprox(self.hullXY, innerPad)
            path = mpl.path.Path(verts)

        if y is None:
            # TODO: assert Nx2 array here!
            xy = x
        else:
            # TODO: assert 2 Nx1 arrays here!
            xy = np.array([x,y]).T

        if len(np.shape(xy)) == 1:
            xy = np.array((xy,))

        return path.contains_points(xy)

    def insideHull(self, x, y=None, z=None, innerPad_xy=None, innerPad_z=None):
        if y is None and z is None:
            # TODO: assert Nx3 array here!
            if len(np.shape(x)) == 1:
                xy = np.array((x[0:2],))
                z = x[2]
            else:
                xy = x[:,0:2]
                z = x[:,2]
        elif y is None and z is not None:
            # TODO: assert Nx2 array here!
            xy = x
            z = z
        elif y is not None and z is not None:
            # TODO: assert 2 Nx1 arrays here!
            xy = np.array([x,y]).T
            z = z
        else: # y is not None and z is None
            raise Exception('Must specify x, y, and z coordinates.')

        if len(np.shape(xy)) == 1:
            xy = np.array((xy,))

        return self.insideHullXY(x=xy, y=None, innerPad=innerPad_xy) & \
                self.insideZ(z, innerPad=innerPad_z)

    def outsideHullXY(self, x, y=None, outerPad=None):
        if outerPad is None:
            path = self.outerHullPath
        elif outerPad == 0:
            path = self.hullPath
        else:
            verts = geometry.expandPolyApprox(self.hullXY, outerPad)
            path = mpl.path.Path(verts)

        if y is None:
            # TODO: assert Nx2 array here!
            xy = x
        else:
            # TODO: assert 2 Nx1 arrays here!
            xy = np.array([x,y]).T

        if len(np.shape(xy)) == 1:
            xy = np.array((xy,))

        return np.logical_not(path.contains_points(xy))

    def outsideHull(self, x, y=None, z=None, outerPad=None):
        if y is None and z is None:
            if len(np.shape(x)) == 1:
                xy = np.array((x[0:2],))
                z = x[2]
            else:
                xy = x[:,0:2]
                z = x[:,2]
        elif y is None and z is not None:
            xy = x
            z = z
        elif y is not None and z is not None:
            xy = np.array([x,y]).T
            z = z
        else: # y is not None and z is None
            raise Exception('Must specify x, y, and z coordinates.')

        if len(np.shape(xy)) == 1:
            xy = np.array((xy,))

        return self.outsideHullXY(x=xy, y=None, outerPad=outerPad) | \
                self.outsideZ(z, outerPad=outerPad)

    def loadDOMXYZ(self):
        ## TODO: test the coordinate loading below; haven't done this yet!!!
        self.all_sd_xyz = np.loadtxt(self.GEOMFILE)

        # IceCube string#,DOM#,x,y,z: Match string numbers, and z-coord at most 500 but
        # listed as 600 here since DOMs are apparently creeping up
        self.IceCube_sd_xyz = self.all_sd_xyz[
            (self.all_sd_xyz[:,0] >= self.ICECUBE_STRINGIND[0]) &
            (self.all_sd_xyz[:,0] <= self.ICECUBE_STRINGIND[-1]) &
            (self.all_sd_xyz[:,4] <= 600)]
        self.IceCube_sd_xyz_dict = {}
        for row in self.IceCube_sd_xyz:
            self.IceCube_sd_xyz_dict[(int(row[0]),int(row[1]))] = row[2:]

        self.DeepCore_sd_xyz = self.all_sd_xyz[(self.all_sd_xyz[:,0] >= self.DEEPCORE_STRINGIND[0]) &
                                               (self.all_sd_xyz[:,0] <= self.DEEPCORE_STRINGIND[-1]) &
                                               (self.all_sd_xyz[:,4] <= 600)]

        self.SD_XYZ = self.all_sd_xyz[(self.all_sd_xyz[:,0] >= self.PINGU_STRINGIND[0]) &
                                      (self.all_sd_xyz[:,0] <= self.PINGU_STRINGIND[-1]) &
                                      (self.all_sd_xyz[:,4] <= 600)]

        all_xy = self.SD_XYZ[:,2:4]
        all_xy = list(set(zip(all_xy[:,0], all_xy[:,1])))
        all_xy.sort()
        self.XY = np.array(all_xy)

        #self.XY = self.SD_XYZ[0::self.N_DOM_PER_STRING,2:4]
        #print np.shape(self.IceCube_sd_xyz), np.shape(self.XY)
        all_z = self.SD_XYZ[:,4]
        all_z = list(set(all_z))
        all_z.sort()
        self.Z = np.array(all_z)
        #self.Z = self.SD_XYZ[0:self.N_DOM_PER_STRING,3]

        #-- IceCube-only (excluding PINGU) strings have string index less than
        #   the starting string index for PINGU strings

        ## TODO: what are the exact string numbers that exclude icetop?
        #  Currently, all these are included here!

        sind = np.where(self.IceCube_sd_xyz[:,0] < self.PINGU_STRINGIND[0])[0]
        all_xy = self.IceCube_sd_xyz[sind,2:4]
        all_xy = list(set(zip(all_xy[:,0], all_xy[:,1])))
        all_xy.sort()
        self.IceCube_xy = np.array(all_xy)
        self.IceCube_sd_xyz_df = pd.DataFrame(self.IceCube_sd_xyz, columns=['string','dom','x','y','z']).set_index(self.uniqueDOMID(self.IceCube_sd_xyz[:,0], self.IceCube_sd_xyz[:,1]))

        return self.IceCube_sd_xyz, self.SD_XYZ

    def PINGUStringXY(self):
        if getattr(self, 'xy', None) is None:
            self.loadDOMXYZ()
        return self.XY

    def IceCubeStringXY(self):
        if getattr(self, 'IceCube_xy', None) is None:
            self.loadDOMXYZ()
        return self.IceCube_xy

    def computeXYHull(self, include_dc):
        if getattr(self, 'xy', None) is None:
            self.loadDOMXYZ()
        if include_dc:
            xy = np.concatenate((self.XY, self.DeepCore_sd_xyz[:,2:4]))
        else:
            xy = self.XY
        self.hull = ConvexHull(xy)
        self.nHullVert = self.hull.nsimplex
        self.hullXY = xy[self.hull.vertices]
        self.hullArea = geometry.polyArea(self.hullXY[:,0], self.hullXY[:,1])
        self.centroidXY = geometry.polyCentroid(self.hullXY)
        self.hullPath = mpl.path.Path(self.hullXY)

    def photonTimeToDOMs(self, event_start_coord, track_stop_coord, n,
                         muon_speed=const.c, light_vac_speed=const.c, t0=0,
                         dom_coords=None, jitter_time=0, debug=False,
                         dbgax=None, dbgstyle=0):
        '''See ~/cowen/quality_of_fit/reports/2014-07-31 for picture of geometry

        event_start_coord : sequence containing x, y, and z coordinates or list thereof [dist]

        track_stop_coord  : sequence containing x, y, and z coordinates or list thereof [dist].
                            NaN is interpreted as NO track present, which avoids all
                            computations specific to track direct light

        n                 : index of refraction of medium

        muon_speed        : speed muon (track) travels at [dist/time], default = c [m/s]

        light_vac_speed   : speed  light travels in vacuum, default = c [m/s]

        t0                : time coordinate of start of event [time], default = 0

        dom_coords        : list of sequences, each containing x, y, and z coordinates of DOMs.
                            Specifying None will load DOM coordinates from the
                            instantiated PINGUGeom object. Default = None.

        jitter_time       : All the muon to start before the beginning or after
                            the end of the track by this much [time]; default = 0.

        debug             : If True, plots the track and direct light paths to the DOMs. Default = False.


        Returns:

        t0+t_gamma_casc   : absolute time of arrival of the cascade's photon,

        t0+t_mu_prop      : absolute time when the muon produces the photon (if muon present; else returns NaN's)

        t0+t_gamm_mu      : absolute time the muon's photon arrives at the DOM (if muon present; else returns NaN's)

        valid_track_mask  : vector with '0' for a hit that is deemed unable to have come from a track of finite
                            length, '1' is deemed possible to have come from the track; if muon not present, returns
                            all 0's

        NOTE:
            All units are mks by default; IceCube, on the other hand, uses ns as the time unit. Since only
        time units differ (units of ns and hence speed units of Gm/s), valid results can be obtained by
        changing both the muon_speed and light_vac_speed parameters to those values in I3 units (DOM
        coordinates remain in meters).
        '''
        if dom_coords is None:
            dom_coords = self.SD_XYZ[:,2:5]
        if muon_speed > light_vac_speed:
            muon_speed = light_vac_speed
        theta_ckv = geometry.cherenkovAngle(n, muon_speed, light_vac_speed=light_vac_speed)
        n_over_c = n/light_vac_speed
        inverse_muon_speed = 1.0/muon_speed

        r_casc = np.array([dom_coords[:,0]-event_start_coord[0],
                           dom_coords[:,1]-event_start_coord[1],
                           dom_coords[:,2]-event_start_coord[2]]).T
        mag_r_casc = np.sqrt(r_casc[:,0]**2 + r_casc[:,1]**2 + r_casc[:,2]**2)

        #-- How much time a photon directly from cascade takes to arrive at DOM
        t_gamma_casc = mag_r_casc * n_over_c

        #if not np.any(np.isnan(track_stop_coord)): # track present
        if np.any(np.logical_not(np.isnan(track_stop_coord))): # track present
            r_track = track_stop_coord - event_start_coord
            mag_r_track = np.sqrt(r_track[0]**2 + r_track[1]**2 + r_track[2]**2)
            ell_min = np.dot(r_casc, r_track) / mag_r_track
            mag_r_min = np.sqrt(mag_r_casc**2 - ell_min**2)
            mag_r_gamma_mu = mag_r_min / np.sin(theta_ckv)
            ell_mu = ell_min - (mag_r_min / np.tan(theta_ckv))

            #-- How much time the muon propagates *before* releasing its Cherenkov photon
            t_mu_prop = ell_mu * inverse_muon_speed

            #-- Check that the muon wouldn't have had to start before the beginning
            #   or after the end of the track, but allowing for this within the
            #   jitter_time.
            #
            # TODO: Implement JP's recommendation of photons coming radially off
            # of muon stopping point, so as to account for muon stopping process,
            # and then there would be no NaNs in the data (unless the user specified
            # no track present)

            #validInd = np.where((t_mu_prop >= -jitter_time) & ((ell_mu-mag_r_track)/light_vac_speed <= jitter_time))[0]
            #valid_track_mask = (t_mu_prop >= -jitter_time) & ((ell_mu-mag_r_track)/light_vac_speed <= jitter_time)
            valid_track_mask = (t_mu_prop >= -jitter_time) & ((t_mu_prop - mag_r_track*inverse_muon_speed) <= jitter_time)
            #invalidInd = np.where((t_mu_prop < -jitter_time) | ((ell_mu-mag_r_track)/light_vac_speed > jitter_time))
            #t_mu_prop[invalidInd] = np.nan

            #-- *After* it's produced, how much time it takes (just) the
            #   Cherenkov photon to arrive at the DOM (i.e., total time since
            #   beginning of event for direct track light to arrive at DOM is
            #   t_mu_prop + t_gamma_mu)
            t_gamma_mu = mag_r_gamma_mu * n_over_c

        else: # no track present
            r_track = 0
            mag_r_track = 0
            ell_min = 0
            mag_r_min = 0
            mag_r_gamma_mu = 0
            ell_mu = 0
            t_mu_prop = np.zeros_like(t_gamma_casc)*np.nan
            t_gamma_mu = np.zeros_like(t_gamma_casc)*np.nan
            valid_track_mask = np.zeros_like(t_gamma_casc)

        if debug:
            from matplotlib import pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            mpl.rc('font', **{'family':'serif', 'weight':'normal', 'size': 10})
            valid_track_ind = np.where(valid_track_mask)[0]
            try:
                muon_stop_coord = np.outer(ell_mu, r_track/mag_r_track) + event_start_coord
                ismuon = True
            except:
                ismuon = False
            ang_betw_r_and_dom = np.arccos(ell_min/mag_r_casc)
            print "DEBUG", "valid_track_ind:\n", valid_track_ind
            print "DEBUG", "theta_ckv:\n", theta_ckv
            print "DEBUG", "mag_r_track:\n", mag_r_track
            print "DEBUG", "mag_r_casc:\n", mag_r_casc
            print "DEBUG", "ell_min:\n", ell_min
            print "DEBUG", "ang_betw_r_and_dom:\n", ang_betw_r_and_dom*180/np.pi
            print "DEBUG", "mag_r_min:\n", mag_r_min
            print "DEBUG", "mag_r_gamma_mu:\n", mag_r_gamma_mu
            print "DEBUG", "ell_mu:\n", ell_mu
            print "DEBUG", "t_gamma_casc:\n", t_gamma_casc
            print "DEBUG", "t_mu_prop:\n", t_mu_prop
            print "DEBUG", "t_gamma_mu:\n", t_gamma_mu
            print "DEBUG", "t_mu_prop+t_gamma_mu:\n", t_mu_prop+t_gamma_mu
            if dbgax is None:
                fig = plt.figure()
                fig.clf()
                ax = fig.add_subplot(111, projection='3d')
                ax.cla()
                newplot = True
            else:
                ax = dbgax
                newplot = False
            if dbgstyle == 0:
                evt_trk_color = (0.0, 1.0, 0.0)
                direct_trk_path_color = 'y'
                muon_stoppt_color = 'k'
                directhitdoms_color = 'c'
                notdirhitdoms_color = 'm'
            elif dbgstyle == 1:
                evt_trk_color = 'b'
                direct_trk_path_color = 'r'
                muon_stoppt_color = 'k'
                directhitdoms_color = 'r'
                notdirhitdoms_color = 'b'
            #ax.scatter(dom_coords[:,0],dom_coords[:,1],dom_coords[:,2],c='r',edgecolors='k')
            dc = dom_coords[valid_track_ind]
            ax.scatter([event_start_coord[0]],
                       [event_start_coord[1]],
                       [event_start_coord[2]],
                       c=evt_trk_color, edgecolors='k', linewidths=1, s=100)
            ax.scatter(dc[:,0], dc[:,1], dc[:,2], c='c', edgecolors='b')
            if ismuon:
                for cn in xrange(len(dc)):
                    ax.plot([muon_stop_coord[valid_track_ind][cn,0],dc[cn,0]],
                            [muon_stop_coord[valid_track_ind][cn,1],dc[cn,1]],
                            [muon_stop_coord[valid_track_ind][cn,2],dc[cn,2]],
                            c=direct_trk_path_color,
                            lw=0.5,
                            alpha=0.5)
                ax.scatter(muon_stop_coord[valid_track_ind][:,0],
                           muon_stop_coord[valid_track_ind][:,1],
                           muon_stop_coord[valid_track_ind][:,2],
                           s=5,
                           c=muon_stoppt_color)
                ax.plot([event_start_coord[0],track_stop_coord[0]],
                        [event_start_coord[1],track_stop_coord[1]],
                        [event_start_coord[2],track_stop_coord[2]], alpha=0.7,
                        c=evt_trk_color)
            ax.scatter(dc[:,0], dc[:,1], dc[:,2], alpha=0.7,
                       c=directhitdoms_color, edgecolors='b')
            dc_not = dom_coords[np.where(np.logical_not(valid_track_mask))[0],:]
            ax.scatter(dc_not[:,0], dc_not[:,1], dc_not[:,2], alpha=0.7,
                       c=notdirhitdoms_color, edgecolors=(0.6,0,0,0.7))
            #ax.scatter(self.SD_XYZ[:,2],
            #           self.SD_XYZ[:,3],
            #           self.SD_XYZ[:,4],
            #           c='k',marker='.',edgecolors=None,linewidths=0,s=2)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.axis('image')
            ax.axis('tight')
            ax.axis('image')
            ax.set_axis_off()
            plt.draw()
            plt.show()

        #-- Save some computational time if t0 = 0 by not adding it to each returned array
        if t0 == 0:
            return t_gamma_casc, t_mu_prop, t_mu_prop+t_gamma_mu, valid_track_mask

        return t0+t_gamma_casc, t0+t_mu_prop, t0+t_mu_prop+t_gamma_mu, valid_track_mask

    def plotIceCube(self, ax_xy, dom_markersize=5, dom_marker='o', dom_color=(0.6,0.6,0)):
        ax_xy.plot(self.IceCube_xy[:,0],self.IceCube_xy[:,1],
                   linestyle='none',
                   marker=dom_marker,
                   markersize=dom_markersize,
                   color=dom_color,
                   label='IceCube strings',
                   alpha=1.0)

    def plotDeepCore(self, ax_xy, dom_markersize=5, dom_marker='s', dom_color=(0,0.6,0)):
        ax_xy.plot(self.DeepCore_sd_xyz[:,2], self.DeepCore_sd_xyz[:,3],
                   linestyle='none',
                   marker=dom_marker,
                   markersize=dom_markersize,
                   color=dom_color,
                   label='DeepCore strings',
                   alpha=1.0)

    def plotPINGU(self, ax_xy, ax_z=None, dom_markersize=5, dom_marker='o',
                  dom_color=(0,0,0)):
        from matplotlib import pyplot as plt
        ax_xy.plot(self.XY[:,0],self.XY[:,1],
                   linestyle='none',
                   marker=dom_marker,
                   markersize=dom_markersize,
                   color=dom_color,
                   label='PINGU strings',
                   alpha=1.0)
        ax_xy.plot(self.centroidXY[0],self.centroidXY[1], 'k',
                   linestyle='none',
                   marker='x',
                   markeredgewidth=1,
                   markersize=5,
                   label='PINGU centroid',
                   alpha=1.0)

        if ax_z is not None:
            xmin = np.min(self.XY[:,0])
            xmax = np.max(self.XY[:,0])
            xz_coords = zip(self.SD_XYZ[:,2], self.SD_XYZ[:,4])
            xz_coords = np.array(sorted(set(xz_coords)))
            z_vertices = [[xmin,self.ZMIN],
                          [xmax,self.ZMIN],
                          [xmax,self.ZMAX],
                          [xmin,self.ZMAX] ]

            ax_z.add_patch(plt.Polygon(z_vertices, closed=True, fill=True,
                                       facecolor='none', edgecolor='b',
                                       linewidth=1))
            ax_z.plot(xz_coords[:,0], xz_coords[:,1],
                      linestyle='none',
                      marker=dom_marker,
                      markersize=dom_markersize,
                      color=dom_color,
                      label='PINGU strings',
                      alpha=1.0)
            ax_z.set_xlabel(r'$x\;(\mathrm{m})$')
            ax_z.set_ylabel(r'$z\;(\mathrm{m})$')


    def plotXYHex(self, ax_xy, drawLabel=False):
        from matplotlib import pyplot as plt
        import matplotlib as mpl
        mpl.rc('font', **{'family':'serif', 'weight':'normal', 'size': 10})
        xy_vertices = [ [self.X0+self.RMAX*np.cos(theta),
                         self.Y0+self.RMAX*np.sin(theta)]
                       for theta in np.arange(0+self.ROT, 2*np.pi+self.ROT,
                                              np.pi/3) ]
        ax_xy.add_patch(plt.Polygon(xy_vertices, closed=True, fill=True,
                                    facecolor='none', edgecolor=(0,0.8,0.8),
                                                                 alpha=0.8,
                                    linewidth=2))
        if drawLabel:
            ax_xy.text(self.X0+self.RMIN*np.sin(self.ROT),
                   self.Y0-self.RMIN*np.cos(self.ROT)+1.5,
                   'Regular hexagon',
                   fontsize=10, ha='left', va='center',
                   rotation=self.ROT*180/np.pi)


    def plotXYHull(self, ax_xy, drawLabel=False):
        from matplotlib import pyplot as plt
        import matplotlib as mpl
        mpl.rc('font', **{'family':'serif', 'weight':'normal', 'size': 10})
        self.hullPoly = plt.Polygon(self.hullXY, closed=True, fill=True,
                                    linewidth=1, facecolor='none',
                                    edgecolor=(0.0,0.0,0.8))
        self.hullPatch = ax_xy.add_patch(self.hullPoly)
        if drawLabel:
            rfact = 0.6
            txt = ax_xy.text(
                self.X0+rfact*self.RMIN*np.sin(self.ROT)+1,
                self.Y0-rfact*self.RMIN*np.cos(self.ROT)+1.5,
                'convex hull',
                fontsize=10, ha='left', va='top', rotation=self.ROT*180/np.pi
            )
            return self.hullPoly, self.hullPatch, txt

        return self.hullPoly, self.hullPatch



class MultiNestLimits:
    def __init__(self, detector, proc_ver, innerPad=0):
        if detector.lower() == 'pingu':
            if proc_ver == 4:
                #-- 5D params
                self.x = np.array([-50, 150])   # m
                self.y = np.array([-135,65])    # m
                self.z = np.array([-550, -150]) # m
                self.t = np.array([-1000, 500])  # ns
                self.ell_mu = np.array([0, 300]) # m

                #-- Included in 8D params
                self.azim = np.array([0, 2*np.pi])
                self.zen = np.array([0, np.pi])
                self.E_casc = np.array([0, 100])

                #-- Included in 10D params

            elif proc_ver == 5:
                #-- 8D params
                self.x = np.array([-50, 150])   # m
                self.y = np.array([-135,65])    # m
                self.z = np.array([-550, -150]) # m
                self.t = np.array([-1000, 500])  # ns
                self.ell_mu = np.array([0, 300]) # m
                self.azim = np.array([0, 2*np.pi])
                self.zen = np.array([0, np.pi])
                self.E_casc = np.array([0, 100])

            else:
                raise Exception('"' + str(proc_ver) + '" is an unknown MultiNest processing version')

        else:
            raise Exception('"' + str(detector) + '" is an unknown detector name')

        self.XMIN = np.min(self.x)
        self.XMAX = np.max(self.x)
        self.YMIN = np.min(self.y)
        self.YMAX = np.max(self.y)
        self.ZMIN = np.min(self.z)
        self.ZMAX = np.max(self.z)

        #-- Ranges
        self.x_r = np.diff(self.x)[0]
        self.y_r = np.diff(self.y)[0]
        self.z_r = np.diff(self.z)[0]
        self.t_r = np.diff(self.t)[0]
        self.ell_mu_r = np.diff(self.ell_mu)[0]

        self.azim_r = np.diff(self.azim)[0]
        self.zen_r = np.diff(self.zen)[0]
        self.E_casc_r = np.diff(self.E_casc)[0]

        #-- Centers (note that azim should be treated specially)
        self.x_c = self.x[0] + self.x_r/2
        self.y_c = self.y[0] + self.y_r/2
        self.z_c = self.z[0] + self.z_r/2
        self.t_c = self.t[0] + self.t_r/2
        self.ell_mu_c = self.ell_mu[0] + self.ell_mu_r/2

        self.azim_c = self.azim[0] + self.azim_r/2
        self.zen_c = self.zen[0] + self.zen_r/2
        self.E_casc_c = self.E_casc[0] + self.E_casc_r/2

        self.vertXY = np.array(
            [ [self.x[0],self.y[0]],
              [self.x[1],self.y[0]],
              [self.x[1],self.y[1]],
              [self.x[0],self.y[1]] ])

        self.path = mpl.path.Path(self.vertXY)

        self.innerPad = innerPad
        if self.innerPad == 0:
            self.vertInside = self.vertXY
            self.pathInside = self.path
        else:
            self.vertInside = geometry.contractPolyApprox(self.vertXY, self.innerPad)
            self.pathInside = mpl.path.Path(self.vertInside)


    def insideZ(self, z, innerPad=None):
        if innerPad is None:
            innerPad = self.innerPad

        zmax = self.ZMAX - innerPad
        zmin = self.ZMIN + innerPad

        return (z < zmax) & (z > zmin)


    def outsideZ(self, z, innerPad=None):
        return np.logical_not(self.insideZ(z, innerPad=innerPad))


    def insideXY(self, x, y=None, innerPad=None):
        if y is None:
            assert x.shape[1] == 2
            xy = x
        else:
            # TODO: assert 2 Nx1 arrays here!
            if len(np.shape(x)) > 1:
                assert x.shape[1] == 1
            if len(np.shape(y)) > 1:
                assert y.shape[1] == 1
            xy = np.array([x,y]).T

        if innerPad is None:
            path = self.pathInside
        else:
            vert = geometry.contractPolyApprox(self.vertXY, innerPad)
            path = mpl.path.Path(vert)

        if len(np.shape(xy)) == 1:
            xy = np.array((xy,))

        return path.contains_points(xy)


    def outsideXY(self, x, y=None, innerPad=None):
        return np.logical_not(self.insideXY(x=x, y=y, innerPad=innerPad))


    def inside(self, x, y=None, z=None, innerPad=None):
        if y is None and z is None:
            # TODO: assert Nx3 array here!
            if len(np.shape(x)) == 1:
                xy = np.array((x[0:2],))
                z = x[2]
            else:
                xy = x[:,0:2]
                z = x[:,2]
        elif y is None and z is not None:
            # TODO: assert Nx2 array here!
            xy = x
            z = z
        elif y is not None and z is not None:
            # TODO: assert 2 Nx1 arrays here!
            xy = np.array([x,y]).T
            z = z
        else: # y is not None and z is None
            raise Exception('Must specify x, y, and z coordinates.')

        if len(np.shape(xy)) == 1:
            xy = np.array((xy,))

        return self.insideXY(x=xy, y=None, innerPad=innerPad) & \
                self.insideZ(z=z, innerPad=innerPad)


    def outside(self, x, y=None, z=None, innerPad=None):
        if y is None and z is None:
            # TODO: assert Nx3 array here!
            if len(np.shape(x)) == 1:
                xy = np.array((x[0:2],))
                z = x[2]
            else:
                xy = x[:,0:2]
                z = x[:,2]
        elif y is None and z is not None:
            # TODO: assert Nx2 array here!
            xy = x
            z = z
        elif y is not None and z is not None:
            # TODO: assert 2 Nx1 arrays here!
            xy = np.array([x,y]).T
            z = z
        else: # y is not None and z is None
            raise Exception('Must specify x, y, and z coordinates.')

        if len(np.shape(xy)) == 1:
            xy = np.array((xy,))

        return self.outsideXY(x=xy, y=None, innerPad=innerPad) | \
                self.outsideZ(z=z, innerPad=innerPad)


    def plotXY(self, ax, drawLabel=False):
        from matplotlib import pyplot as plt
        import matplotlib as mpl
        mpl.rc('font', **{'family':'serif', 'weight':'normal', 'size': 10})
        ax.add_patch(plt.Polygon(self.vertXY, closed=True, fill=True,
                                 facecolor='none', edgecolor=(0.6,0.0,0.0),
                                 alpha=1.0, linewidth=2))
        if drawLabel:
            ax.text(self.x[0]+2, self.y[0]-2,
                    'MultiNest limits',
                    fontsize=10, ha='left', va='top')


def stopCoord(x, y, z, r, zen, az):
    xrel, yrel, zrel = geometry.sph2cart(r=r, theta=np.pi-zen, phi=az+np.pi)
    xrel[np.isnan(xrel)] = 0
    yrel[np.isnan(yrel)] = 0
    zrel[np.isnan(zrel)] = 0
    return x + xrel, y + yrel, z + zrel


def attachStopCoord(ss):
    ss['MNr1_8D_Track__stop_x'], \
            ss['MNr1_8D_Track__stop_y'], \
            ss['MNr1_8D_Track__stop_z'] = stopCoord(
                x=ss['MNr1_8D_Neutrino__x'],
                y=ss['MNr1_8D_Neutrino__y'],
                z=ss['MNr1_8D_Neutrino__z'],
                r=ss['MNr1_8D_Track__length'],
                zen=ss['MNr1_8D_Track__zenith'],
                az=ss['MNr1_8D_Track__azimuth']
            )
    return ss


class Cut(object):
    def __init__(self):
        pass
    def apply(self, ss):
        pass
    def plot(self):
        pass

def cylinderCut_ss(x0, y0, z0, r, dz,
                   vx_name='MNr1_8D_Neutrino__x',
                   vy_name='MNr1_8D_Neutrino__y',
                   vz_name='MNr1_8D_Neutrino__z',
                   calculate_vol=False):
    z1 = z0 + dz
    r2 = r**2
    def cut(ss):
        vr2 = (ss[vx_name]-x0)**2 + (ss[vy_name]-y0)**2
        return (ss[vz_name] >= z0) & (ss[vz_name] <= z1) & (vr2 <= r2)
    if calculate_vol:
        return cut, (np.pi * r2 * dz)
    return cut

def cylinderCut(x0, y0, z0, r, dz, vx, vy, vz):
    z1 = z0 + dz
    r2 = r**2
    vr2 = (vx-x0)**2 + (vy-y0)**2
    return (vz >= z0) & (vz <= z1) & (vr2 <= r2)

def rect3DCut_ss(x0, x1, y0, y1, z0, z1,
                 vx_name='MNr1_8D_Neutrino__x',
                 vy_name='MNr1_8D_Neutrino__y',
                 vz_name='MNr1_8D_Neutrino__z'):
    def cut(ss):
        return (ss[vx_name] >= x0) & (ss[vx_name] <= x1) & \
                (ss[vy_name] >= y0) & (ss[vy_name] <= y1) & \
                (ss[vz_name] >= z0) & (ss[vz_name] <= z1)
    return cut

def rect3DCut(x0, x1, y0, y1, z0, z1, vx, vy, vz):
    return (vx >= x0) & (vx <= x1) & (vy >= y0) & (vy <= y1) & (vz >= z0) & (vz <= z1)

#def hullCut(pg, inner_pad_xy, inner_pad_z, vx, vy, vz):
#    return pg.insideHull(
#        vx, vy, vz,
#                         innerPad_xy=inner_pad_xy,
#                         innerPad_z=inner_pad_z)
#    return cut

class StandardCuts(Cut):
    def __init__(self, geom_ver, proc_ver):
        self.pg = PINGUGeom(version=geom_ver)
        if proc_ver in [4, 5]:
            self.step2 = cylinderCut(
                x0=pg.CUT_X0,
                y0=pg.CUT_Y0,
                z0=pg.CUT_ZMIN,
                r=85,
                dz=pg.CUT_ZMAX-pg.CUT_ZMIN,
                vx_name='MNr1_8D_Neutrino__x',
                vy_name='MNr1_8D_Neutrino__y',
                vz_name='MNr1_8D_Neutrino__z'
            )

class Cuts:
    def __init__(self, version):
        self.version = version
        self.all_z_range = {
                26: [-500, -180],
                22: [-500, -180],
                24: [-500, -180],
                30: [-500, -180],
                27: [-500, -180],
                15: [-500, -180],
                17: [-500, -180],
                29: [-500, -180],
                23: [-500, -180],
                25: [-500, -180],
                31: [-500, -180],
                28: [-500, -180],
                36: [-500, -180],
                38: [-500, -180],
                39: [-500, -180],
                }
        self.all_xy_center = {
                26: [45, -38],
                22: [50, -35],
                24: [50, -35],
                30: [50, -35],
                27: [44, -40],
                15: [50, -50],
                17: [50, -50],
                29: [50, -35],
                23: [50, -50],
                25: [50, -50],
                31: [50, -50],
                28: [50, -45],
                36: [50, -35],
                38: [50, -35],
                39: [50, -35],
                }
        self.all_xy_radius = {
                26: 25,
                22: 55,
                24: 55,
                30: 55,
                27: 60,
                15: 75,
                17: 75,
                29: 90,
                23: 90,
                25: 90,
                31: 90,
                28: 175,
                36: 100,
                38: 100,
                39: 100,
                }
        self.all_xy_radius_step2 = {
                36: 85,
                38: 85,
                39: 85,
                }
        if version not in self.all_z_range:
            raise Exception('"' + str(version) + '" is an unknown cut version')

        self.z_range = self.all_z_range[version]
        self.xy_center = self.all_xy_center[version]
        self.xy_radius_step1 = self.all_xy_radius[version]
        self.xy_radius_rough = self.all_xy_radius[version] + 10
        self.cont_vol_step1 = np.pi * self.xy_radius_step1**2 * self.z_range[1] - self.z_range[0]
        self.cont_vol_rough = np.pi * self.xy_radius_rough**2 * self.z_range[1] - self.z_range[0]
        if version in self.all_xy_radius_step2:
            self.xy_radius_step2 = self.all_xy_radius_step2[version]
        else:
            self.xy_radius_step2 = self.all_xy_radius[version]
        self.cont_vol_step2 = np.pi * self.xy_radius_step2**2 * self.z_range[1] - self.z_range[0]

    def zCut(self, z):
        return (z > self.z_range[0]) & (z < self.z_range[1])

    def rCut(self, x, y, step):
        if step == 1:
            return ((x-self.xy_center[0])*(x-self.xy_center[0]) + (y-xy_center[1])*(y-xy_center[1])) < self.xy_radius_step1*self.xy_radius_step1
        elif step == 1.5:
            return ((x-self.xy_center[0])*(x-self.xy_center[0]) + (y-xy_center[1])*(y-xy_center[1])) < self.xy_radius_rough*self.xy_radius_rough
        elif step == 2:
            return ((x-self.xy_center[0])*(x-self.xy_center[0]) + (y-xy_center[1])*(y-xy_center[1])) < self.xy_radius_step2*self.xy_radius_step2
        else:
            raise ValueError('Unrecognized step: %s' % step)

    def cylCut(self, x, y, z, step):
        return self.rCut(x, y, step=step) & self.zCut(z)

    def plotCutXY(self, step, ax, drawLabel=False):
        from matplotlib import pyplot as plt
        import matplotlib as mpl
        mpl.rc('font', **{'family':'serif', 'weight':'normal', 'size': 10})
        if step == 1:
            rad = self.xy_radius_step1
            self.step1circ = plt.Circle(self.xy_center, self.xy_radius_step1,
                                        facecolor='none',
                                        linewidth=2,
                                        edgecolor=(0.2,1.0,0.2))
            ax.add_artist(self.step1circ)
            if drawLabel:
                ax.text(self.xy_center[0],
                       self.xy_center[1] - self.xy_radius_step1+2,
                       'Cut step 1', fontsize=10, ha='center', va='bottom')


        elif step == 1.5: # (rough cut)
            return
            self.roughCirc = plt.Circle(self.xy_center, self.xy_radius_rough,
                                        facecolor='none',
                                        linewidth=2, #linestyle='--',
                                        edgecolor=(0.4,0.8,0.0))
            ax.add_artist(self.roughCirc)
            if drawLabel:
                ax.text(self.xy_center[0],
                       self.xy_center[1] - self.xy_radius_rough+2,
                       'Rough cut', fontsize=10, ha='center', va='bottom')

        elif step == 2:
            self.step2circ = plt.Circle(self.xy_center, self.xy_radius_step2,
                                        facecolor='none',
                                        linewidth=2,
                                        edgecolor=(0.4,0.4,0.0))
            ax.add_artist(self.step2circ)
            if drawLabel:
                ax.text(self.xy_center[0],
                       self.xy_center[1] - self.xy_radius_step2+2,
                       'Cut step 2', fontsize=10, ha='center', va='bottom')

        else:
            raise Exception('Invalid step: "' + str(step) + '"')


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    mpl.rc('font', **{'family':'serif', 'weight':'normal', 'size': 10})
    SAVE_FIGURES = True

    N = 1000000
    innerPad = 0*-25
    outerPad = 0*25
    innerPadMNLimits = 0
    eventColor = (0.8, 0.8, 0.8)
    eventMS = 2
    figsize_xy = (8,8)
    figsize_z = (8,10)

    PG = PINGUGeom(version=36, include_dc=True, innerPad=innerPad, outerPad=outerPad)
    CT = Cuts(version=36)
    MN = MultiNestLimits(detector='pingu', proc_ver=5, innerPad=0)

    x = (np.random.random(N)-0.5)*MN.x_r*1.5 + MN.x_c
    y = (np.random.random(N)-0.5)*MN.y_r*1.5 + MN.y_c
    #z = (np.random.random(N)-0.5)*MN.z_r*1.5 + MN.z_c
    z = (np.random.random(N)-0.5)*MN.z_r/2 + MN.z_c

    vertWithinMN = geometry.contractPolyApprox(MN.vertXY, innerPadMNLimits)
    pathWithinMN = mpl.path.Path(vertWithinMN)

    insideInds = np.where(PG.insideHull(x, y, z,
                                        innerPad_xy=innerPad,
                                        innerPad_z=innerPad))
    outsideInds = np.where(PG.outsideHull(x,y,z, outerPad=outerPad) &
                           MN.insideXY(x, y, innerPad=innerPadMNLimits))

    #insideInds = np.where(PG.insideHullXY(x,y, innerPad=innerPad))
    #outsideInds = np.where(PG.outsideHullXY(x,y, outerPad=outerPad))

    f200 = plt.figure(200, figsize=figsize_xy)
    f200.clf()
    ax200 = f200.add_subplot(111)
    a = ax200
    f210 = plt.figure(210, figsize=figsize_z)
    f210.clf()
    ax_z = f210.add_subplot(111)

    MN.plotXY(a, drawLabel=True)
    CT.plotCutXY(step=1, ax=a, drawLabel=True)
    CT.plotCutXY(step=1.5, ax=a, drawLabel=True)
    CT.plotCutXY(step=2, ax=a, drawLabel=True)
    a.plot(x[insideInds], y[insideInds], '.',
           ms=eventMS,
           color=eventColor,
           #label='"inside" PINGU (' + str(-innerPad) + ' m border)',
           alpha=1.0, markersize=2, zorder=0.75)
    PG.plotIceCube(a, dom_markersize=5)
    PG.plotDeepCore(a, dom_markersize=5)
    #PG.plotXYHex(a, drawLabel=True)
    PG.plotXYHull(a, drawLabel=False)
    PG.plotPINGU(ax_xy=a, ax_z=ax_z, dom_markersize=5)

    leg200 = a.legend(loc='upper center', numpoints=1, framealpha=1.0,
                      bbox_to_anchor=(0.5,1.135), fancybox=False, shadow=False,
                      ncol=2)
    a.set_xlabel(r'$x$ (m)')
    a.set_ylabel(r'$y$ (m)')
    a.axis('equal')
    a.set_xlim(MN.x)
    a.set_ylim((MN.y-MN.y_c)*1.50+MN.y_c)
    plt.draw()

    #-- Plot OUTSIDE points
    f201 = plt.figure(201, figsize=figsize_xy)
    f201.clf()
    ax201 = f201.add_subplot(111)
    a = ax201
    MN.plotXY(a, drawLabel=True)
    CT.plotCutXY(step=1, ax=a, drawLabel=True)
    CT.plotCutXY(step=1.5, ax=a, drawLabel=True)
    CT.plotCutXY(step=2, ax=a, drawLabel=True)
    a.plot(x[outsideInds], y[outsideInds], '.',
           ms=eventMS,
           color=eventColor,
           #label='"outside" PINGU (' + str(outerPad) +
           #      ' m pad) \n& within MN limits (' +
           #      str(innerPadMNLimits) + ' m pad)',
           alpha=1.0, markersize=2, zorder=0.75)
    PG.plotIceCube(a, dom_markersize=5)
    PG.plotDeepCore(a, dom_markersize=5)
    #PG.plotXYHex(a, drawLabel=True)
    PG.plotXYHull(a, drawLabel=False)
    PG.plotPINGU(a, dom_markersize=5)
    a.plot(CT.xy_center[0], CT.xy_center[1], 'k',
           linestyle='none',
           marker='+',
           markeredgewidth=1,
           mfc='none',
           markersize=5,
           label='Circular cuts\' center',
           alpha=1.0)

    leg201 = a.legend(loc='upper center', numpoints=1, framealpha=1.0,
                      bbox_to_anchor=(0.5,1.135), fancybox=True, shadow=True,
                      ncol=2)
    a.set_xlabel(r'$x$ (m)')
    a.set_ylabel(r'$y$ (m)')
    a.axis('equal')
    #a.set_xlim(MN.x)
    #a.set_ylim((MN.y-MN.y_c)*1.50+MN.y_c)
    a.set_xlim((-150, 250))
    a.set_ylim((MN.y-MN.y_c)*2.50+MN.y_c)

    if SAVE_FIGURES:
        f200.savefig(os.path.join(os.path.expanduser('~'),'cowen', 'scripts', 'points_inside_verification.pdf'))
        f200.savefig(os.path.join(os.path.expanduser('~'),'cowen', 'scripts', 'points_inside_verification.png'), dpi=600)
        f201.savefig(os.path.join(os.path.expanduser('~'),'cowen', 'scripts', 'points_outside_verification.pdf'))
        f201.savefig(os.path.join(os.path.expanduser('~'),'cowen', 'scripts', 'points_outside_verification.png'), dpi=600)
    plt.draw()
    plt.show()

    ##-- Test distances
    #pg = PINGUGeom(version=36)
    #x=-35
    #y=-30
    #f=figure(1)
    #clf()
    #ax=f.add_subplot(111)
    #pg.plotPINGU(ax_xy=ax)
    #pg.plotXYHull(ax_xy=ax)
    #plot(x,y,'ro')
    #print pg.xyDistFromHullEdge(x,y)
