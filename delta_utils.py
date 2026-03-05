#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Thomas Flayols, feb 2022

from math import atan2, cos, sin, sqrt
import matplotlib.pyplot as plt
import numpy as np

# System size         #        /\x,y
l1 = 0.06             #       /  \
l2 = 0.125            #  ^   /    \<-l2
d = 0.130 / 2         #  |   \_dd_/<-l1
                      #  |   q1  q2
                      # Y|
                      #  |____>
                      #    X

_EPS = 1e-12


def _nan_vec():
    return np.array([np.nan, np.nan], dtype=float)


def ik_serial(x, y, positive=True):
    c2 = (x * x + y * y - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if c2 < -1.0 - _EPS or c2 > 1.0 + _EPS:
        return (np.nan, np.nan)

    c2 = np.clip(c2, -1.0, 1.0)
    s2 = sqrt(max(0.0, 1.0 - c2 * c2))
    if not positive:
        s2 = -s2

    q2 = atan2(s2, c2)
    q1 = atan2(y, x) - atan2(l2 * s2, l1 + l2 * c2)
    return (q1, q2)

def plot_serial(q1,q2,x0=0.0):
    plt.axis('equal')
    xa = cos(q1) * l1 + x0
    ya = sin(q1) * l1
    xb = xa + cos(q1+q2) * l2
    yb = ya + sin(q1+q2) * l2
    plt.plot([x0,xa,xb] , [0,ya,yb],"o-")

def ik_delta(p):
    x = p[0]
    y = p[1]
    # Keep IK branch continuous: no solution below the base line.
    if y < 0.0:
        return _nan_vec()

    # Fixed IK branch pair for continuous, outward elbows:
    # left arm -> negative serial branch, right arm -> positive serial branch.
    q0, _ = ik_serial(x + d, y, False)  # left base at -d
    q1, _ = ik_serial(x - d, y, True)   # right base at +d

    if np.isnan(q0) or np.isnan(q1):
        return _nan_vec()
    return np.array([q0, q1], dtype=float)
    
def plot_delta(q,line="o-"):
    plt.axis('equal')
    p = fk_delta(q)
    if np.isnan(p).any():
        return
    x, y = p

    xl = cos(q[0]) * l1 - d
    yl = sin(q[0]) * l1
    
    xr = cos(q[1]) * l1 + d
    yr = sin(q[1]) * l1
    plt.plot([-d,xl,x] , [0,yl,y],line,color="red")
    plt.plot([+d,xr,x] , [0,yr,y],line,color="blue")

def get_intersections(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1
    dx = x1 - x0
    dy = y1 - y0
    d2 = dx * dx + dy * dy
    if d2 <= _EPS:
        return np.full((2, 2), np.nan, dtype=float)

    d01 = sqrt(d2)
    if d01 > r0 + r1 + _EPS:
        return np.full((2, 2), np.nan, dtype=float)
    if d01 < abs(r0 - r1) - _EPS:
        return np.full((2, 2), np.nan, dtype=float)

    a = (r0 * r0 - r1 * r1 + d2) / (2.0 * d01)
    h2 = r0 * r0 - a * a
    if h2 < -_EPS:
        return np.full((2, 2), np.nan, dtype=float)

    h = sqrt(max(0.0, h2))
    x2 = x0 + a * dx / d01
    y2 = y0 + a * dy / d01

    rx = -dy * h / d01
    ry = dx * h / d01
    return np.array([[x2 + rx, y2 + ry], [x2 - rx, y2 - ry]], dtype=float)


def fk_delta(q,positive=True):
    x0 = -d + l1 * cos(q[0])
    y0 = l1 * sin(q[0])
    x1 = d + l1 * cos(q[1])
    y1 = l1 * sin(q[1])
    intersections = get_intersections(x0, y0, l2, x1, y1, l2)

    if np.isnan(intersections).any():
        return _nan_vec()

    p0 = intersections[0]
    p1 = intersections[1]
    if positive:
        return p0 if p0[1] >= p1[1] else p1
    return p0 if p0[1] <= p1[1] else p1

def J(q):
    eps = 1e-6
    base = fk_delta(q)
    if np.isnan(base).any():
        return np.full((2, 2), np.nan, dtype=float)

    jac = np.empty((2, 2), dtype=float)
    dq0 = eps * np.array([1.0, 0.0])
    dq1 = eps * np.array([0.0, 1.0])

    f0 = fk_delta(q + dq0)
    f1 = fk_delta(q + dq1)
    if np.isnan(f0).any() or np.isnan(f1).any():
        return np.full((2, 2), np.nan, dtype=float)

    jac[:, 0] = (f0 - base) / eps
    jac[:, 1] = (f1 - base) / eps
    return jac
    
def plot_box():
    plt.plot([-0.105,0.105,0.105,-0.105,-0.105],[0.04,0.04,-0.04,-0.04,0.04])
    
    
