#!/usr/bin/env python

import numpy as np
import scipy.optimize

# https://pypi.python.org/pypi/transforms3d
import transforms3d

def make_transform(tx, ty, tz, rotz):
    ''' Creates a 4x4 rigid transform matrix with
    translation: tx, ty, tz
    rotation: rotz radians around z axis
    '''
    rot = transforms3d.axangles.axangle2mat([0, 0, 1], rotz)
    return transforms3d.affines.compose([tx, ty, tz], rot, [1, 1, 1])

def transform_points(points, tf):
    ''' Input matrix of N points (one per column) 3xN
    Outputs points in the same format '''
    points_h = np.vstack((points, np.ones((1, points.shape[1]))))
    tpoints = np.matmul(tf, points_h)
    return tpoints[0:3, :] / tpoints[3, :]

def angle_between(v1, v2):
    ''' Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    '''
    v1_u = transforms3d.utils.normalized_vector(v1)
    v2_u = transforms3d.utils.normalized_vector(v2)

    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def angle_between_vectorized(v1, v2):
    ''' Vectorized version of angle_between: v1 and v2 are 2D matrices, one vector per row.
    Returns 1d array, one element per row
    '''
    v1_u = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    v2_u = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
    dot = np.einsum("ij,ij->i", v1_u, v2_u) # dot product for each row
    return np.arccos(np.clip(dot, -1.0, 1.0))

def error_func(p, qc, qv, tx, ty, tz, rotz):
    ''' Transform points p using tx, ty, tz, rotz.
    For each transformed point tp, compute the angle between:
    - the direction joining qc and tp
    - the direction qv '''
    tf = make_transform(tx, ty, tz, rotz)
    tp = transform_points(p, tf)
    return list(angle_between_vectorized(qv.T, (tp - qc).T))

def estimate_pose(p, qc, qv, x0):
    ''' Given points in robot frame (p) and rays in human frame (qc, qv), find
    transformation parameters from human frame to robot frame that minimize the
    residual, using starting x0 as the initial solution '''

    def f(x):
        return np.mean(error_func(p, qc, qv, *x)) + max(0.0, np.linalg.norm(x[:3]) - 7.0)# + max(0.0, np.abs(x[2]) - 1.0)

    res = scipy.optimize.minimize(f, x0)
    maxerr = np.max(error_func(p, qc, qv, *res.x)) + max(0.0, np.linalg.norm(res.x[:3]) - 7.0)

    return res, maxerr

def estimate_pose_no_constraints(p, qc, qv, x0):
    ''' Given points in robot frame (p) and rays in human frame (qc, qv), find
    transformation parameters from human frame to robot frame that minimize the
    residual, using starting x0 as the initial solution '''

    def f(x):
        return np.mean(error_func(p, qc, qv, *x))

    res = scipy.optimize.minimize(f, x0)

    maxerr = np.max(error_func(p, qc, qv, *res.x))

    return res, maxerr