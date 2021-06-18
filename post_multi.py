import numpy as np
import pandas as pd

import sys
import pickle
import random
import os
import time
from datetime import datetime
from pathlib import Path
import glob
import json
from tqdm.notebook import tqdm

import matplotlib.path as mpltPath
from scipy.spatial import distance
from scipy.optimize import minimize
from scipy.interpolate import interp1d

def do_foo(site_id, num):

    data = {1: num, 2: num}

    return {site_id: data}


DELAY_OFFSET = 500  # msecs
V_LIMIT = 0.0015  # m/msec
INVALID_RANGE = 0.5 # invalidity distance in fraction of trajectory extent
SCALE_XY0 = 0.01
T_CUT = 130000  # max segement duration in mseconds

def optimize_trajectory_motion(motion_x, motion_y, motion_t,
                               coarse_x, coarse_y, coarse_t,
                               leaked_record, step_range, angle_range, angle0_range,
                               verbose=False):  # +-step_range in fraction of step length; +-angle_range in fraction of Pi

    num_elements = motion_t.shape[0]
    trajectory_extent = ((coarse_x.max() - coarse_x.min()) ** 2 + (coarse_x.max() - coarse_x.min()) ** 2) ** 0.5

    time_deltas = np.diff(motion_t)  # size "n-1"
    motion_steps_lengths_input = ((motion_x ** 2 + motion_y ** 2) ** 0.5)[1:]  # size "n-1"
    motion_angles_input = np.arctan2(motion_y[1:], motion_x[1:])  # size "n-1"
    # print(num_elements)

    ################  INITIAL/BOUNDARIES VALUES   ##################################
    step_lengths_ini = np.amin(np.vstack([time_deltas * V_LIMIT, motion_steps_lengths_input]), axis=0)
    angles_ini = motion_angles_input

    step_lengths_min = np.amin(np.vstack([time_deltas * V_LIMIT * 0.99, motion_steps_lengths_input * (1 - step_range)]),
                               axis=0)
    step_lengths_max = np.amin(np.vstack([time_deltas * V_LIMIT * 1.00, motion_steps_lengths_input * (1 + step_range)]),
                               axis=0)  # not more than maximum speed nor allowed variation of step length
    angles_min = motion_angles_input - np.pi * angle_range
    angles_max = motion_angles_input + np.pi * angle_range

    angle0_ini = np.array([0.0])

    # if non sequential=>negative-invalid delay/negative-invalid position point/delay is too long => take no limits, i.e. something large e.g. 1e6 = 1000s ~ 1500meters
    # make rough estimate based on median of coarse_data in case of very long/invalid delays/invaliude start-end points(-1m value)

    startpoint_invalid = False
    if leaked_record["start_x"] > 0 and leaked_record["start_delay"] > 0 and (
            V_LIMIT * leaked_record["start_delay"]) < trajectory_extent * INVALID_RANGE:
        _start_delay = leaked_record["start_delay"] - DELAY_OFFSET if leaked_record[
                                                                          "start_delay"] > DELAY_OFFSET else 100
    else:
        startpoint_invalid = True
        _start_delay = 1e6

    endpoint_invalid = False
    if leaked_record["end_x"] > 0 and leaked_record["end_delay"] > 0 and (
            V_LIMIT * leaked_record["end_delay"]) < trajectory_extent * INVALID_RANGE:
        _end_delay = leaked_record["end_delay"] - DELAY_OFFSET if leaked_record["end_delay"] > DELAY_OFFSET else 100

    else:
        endpoint_invalid = True
        _end_delay = 1e6

    initial_start = np.array([coarse_x[0], coarse_y[0]])
    initial_end = np.array([coarse_x[-1], coarse_y[-1]])

    x_start_min = leaked_record["start_x"] - _start_delay * V_LIMIT
    x_start_max = leaked_record["start_x"] + _start_delay * V_LIMIT
    x_end_min = leaked_record["end_x"] - _end_delay * V_LIMIT
    x_end_max = leaked_record["end_x"] + _end_delay * V_LIMIT
    y_start_min = leaked_record["start_y"] - _start_delay * V_LIMIT
    y_start_max = leaked_record["start_y"] + _start_delay * V_LIMIT
    y_end_min = leaked_record["end_y"] - _end_delay * V_LIMIT
    y_end_max = leaked_record["end_y"] + _end_delay * V_LIMIT

    bounds_angle0 = [(-angle0_range * np.pi, angle0_range * np.pi)]
    bounds_start = [(0, None), (0, None)]
    # bounds_start = [(x_start_min, x_start_max), (y_start_min, y_start_max)]  # 2x pairs
    bounds_end = [(0, None), (0, None)]
    # bounds_end = [(x_end_min, x_end_max), (y_end_min, y_end_max)] # 2x pairs
    bounds_steps = [(min_el, max_el) for min_el, max_el in zip(step_lengths_min, step_lengths_max)]  # (n-1)x pairs
    bounds_angles = [(min_el, max_el) for min_el, max_el in zip(angles_min, angles_max)]  # (n-1)x pairs

    if endpoint_invalid:
        initial_guess = np.concatenate([step_lengths_ini, angles_ini, angle0_ini, initial_start])  # 1+(2*n+2)x
        bounds = bounds_steps + bounds_angles + bounds_angle0 + bounds_start  # 1+(2*n+2)x pairs
    else:
        initial_guess = np.concatenate(
            [step_lengths_ini, angles_ini, angle0_ini, initial_start, initial_end])  # 1+(2*n+2)x
        bounds = bounds_steps + bounds_angles + bounds_angle0 + bounds_start + bounds_end  # 1+(2*n+2)x pairs

    if verbose:
        print("----------------------------------------------------------")
        print("invalid distance", trajectory_extent * INVALID_RANGE)
        print(f"invalid start/end: {startpoint_invalid}/{endpoint_invalid}")
        print(
            f'leaked_start/end:  {leaked_record["start_x"]}-{leaked_record["start_y"]}/{leaked_record["end_x"]}-{leaked_record["end_y"]}')
        print(f"_start/end_delay: {_start_delay}/{_end_delay} ")
        print(f"initial_start/end:  {initial_start}/{initial_end}")
        print(f"bounds_start/end: {bounds_start}/{bounds_end}")
        print("----------------------------------------------------------")
    ###############################################################################
    d_ind_a = 1  # indices shift due to alpha0

    def loss(params):
        # print(params.shape)
        _d_xs = np.cumsum(params[: num_elements - 1] * np.cos(
            params[2 * num_elements - 2] + params[num_elements - 1: 2 * num_elements - 2]))
        _d_ys = np.cumsum(params[: num_elements - 1] * np.sin(
            params[2 * num_elements - 2] + params[num_elements - 1: 2 * num_elements - 2]))
        _xs = np.hstack([params[2 * num_elements - 2 + d_ind_a + 0],
                         params[2 * num_elements - 2 + d_ind_a + 0] + _d_xs])
        _ys = np.hstack([params[2 * num_elements - 2 + d_ind_a + 1],
                         params[2 * num_elements - 2 + d_ind_a + 1] + _d_ys])

        if not endpoint_invalid:
            _xs = _xs - (_xs[-1] - params[-2]) / 1
            _ys = _ys - (_ys[-1] - params[-1]) / 1

        _loss = np.sum(((coarse_x - _xs) ** 2 + (coarse_y - _ys) ** 2) ** 0.5) / num_elements  # MAE
        return _loss

    results = minimize(fun=loss,
                       x0=initial_guess,
                       bounds=bounds,
                       options={'maxcor': 50, 'ftol': 1e-09, 'gtol': 1e-09, 'maxfun': 40000, 'maxiter': 40000,
                                'maxls': 50},
                       method="L-BFGS-B")
    params = results.x

    _d_xs = np.cumsum(params[: num_elements - 1] * np.cos(
        params[2 * num_elements - 2] + params[num_elements - 1: 2 * num_elements - 2]))
    _d_ys = np.cumsum(params[: num_elements - 1] * np.sin(
        params[2 * num_elements - 2] + params[num_elements - 1: 2 * num_elements - 2]))
    _xs = np.hstack([params[2 * num_elements - 2 + d_ind_a + 0],
                     params[2 * num_elements - 2 + d_ind_a + 0] + _d_xs])
    _ys = np.hstack([params[2 * num_elements - 2 + d_ind_a + 1],
                     params[2 * num_elements - 2 + d_ind_a + 1] + _d_ys])

    if not endpoint_invalid:
        _xs = _xs - (_xs[-1] - params[-2]) / 1
        _ys = _ys - (_ys[-1] - params[-1]) / 1

    return _xs, _ys, results.fun

def optimize_trajectory_fs_v1(motion_x, motion_y, motion_t,
                              fs_points,
                              leaked_record,
                              snap_range,  # +-snap/start_point shift range in fraction of x/y bounding box (max-min)
                              step_range, angle_range,
                              angle0_range):  # +-step_range in fraction of step length; +-angle_range in fraction of Pi

    num_elements = motion_t.shape[0]
    trajectory_extent = ((motion_x.max() - motion_x.min()) ** 2 + (motion_y.max() - motion_y.min()) ** 2) ** 0.5
    trajectory_extent_x = abs(motion_x.max() - motion_x.min())
    trajectory_extent_y = abs(motion_y.max() - motion_y.min())

    snap_distance = snap_range * trajectory_extent
    snap_distance_x = snap_range * trajectory_extent_x
    snap_distance_y = snap_range * trajectory_extent_y

    time_deltas = np.diff(motion_t)  # size "n-1"
    motion_dx = np.diff(motion_x)
    motion_dy = np.diff(motion_y)

    motion_steps_lengths_input = (motion_dx ** 2 + motion_dy ** 2) ** 0.5
    motion_angles_input = np.arctan2(motion_dy, motion_dx)

    d_ind_a = 1  # index shift due to angle0 parameter
    # print(num_elements)

    ################  INITIAL/BOUNDARIES VALUES   ##################################
    step_lengths_ini = np.amin(np.vstack([time_deltas * V_LIMIT, motion_steps_lengths_input]), axis=0)
    angles_ini = motion_angles_input

    step_lengths_min = np.amin(np.vstack([time_deltas * V_LIMIT * 0.99, motion_steps_lengths_input * (1 - step_range)]),
                               axis=0)
    step_lengths_max = np.amin(np.vstack([time_deltas * V_LIMIT * 1.00, motion_steps_lengths_input * (1 + step_range)]),
                               axis=0)  # not more than maximum speed nor allowed variation of step length
    angles_min = motion_angles_input - np.pi * angle_range
    angles_max = motion_angles_input + np.pi * angle_range

    angle0_ini = np.array([0.0])

    initial_start = np.array([motion_x[0], motion_y[0]])

    x_start_min = initial_start[0] - snap_distance_x
    x_start_max = initial_start[0] + snap_distance_x
    y_start_min = initial_start[1] - snap_distance_y
    y_start_max = initial_start[1] + snap_distance_y

    bounds_angle0 = [(-angle0_range * np.pi, angle0_range * np.pi)]
    bounds_start = [(0, None), (0, None)]  # [(x_start_min, x_start_max), (y_start_min, y_start_max)]  # 2x pairs
    bounds_steps = [(min_el, max_el) for min_el, max_el in zip(step_lengths_min, step_lengths_max)]  # (n-1)x pairs
    bounds_angles = [(min_el, max_el) for min_el, max_el in zip(angles_min, angles_max)]  # (n-1)x pairs

    initial_guess = np.concatenate([step_lengths_ini, angles_ini, angle0_ini, initial_start])  # (2*n+2)x
    bounds = bounds_steps + bounds_angles + bounds_angle0 + bounds_start  # (2*n+2)x pairs
    ###############################################################################
    #########################  SNAPPING POINTS   ##################################
    snap_points = []

    for i_step, _ in enumerate(motion_x):
        _distances = ((fs_points[:, 0] - motion_x[i_step]) ** 2 + (fs_points[:, 1] - motion_y[i_step]) ** 2) ** 0.5
        _snap_points = fs_points[_distances <= snap_distance]

        if len(_snap_points) > 0:
            snap_points.append(_snap_points)

    snap_points = np.concatenate(snap_points)
    snap_points = np.array(list(set(map(tuple, snap_points))))

    # print("snap_distance", snap_distance)
    # print("snap_points.shape", snap_points.shape)
    ##################################################################################
    ######################### OPTIMIZE ###############################################
    if len(snap_points) > 0:  # some points are within snapping range

        def loss(params):
            # print(params.shape)
            _d_xs = np.cumsum(params[: num_elements - 1] * np.cos(
                params[2 * num_elements - 2] + params[num_elements - 1: 2 * num_elements - 2]))
            _d_ys = np.cumsum(params[: num_elements - 1] * np.sin(
                params[2 * num_elements - 2] + params[num_elements - 1: 2 * num_elements - 2]))
            _xs = np.hstack([params[2 * num_elements - 2 + d_ind_a + 0],
                             params[2 * num_elements - 2 + d_ind_a + 0] + _d_xs])
            _ys = np.hstack([params[2 * num_elements - 2 + d_ind_a + 1],
                             params[2 * num_elements - 2 + d_ind_a + 1] + _d_ys])

            distances2 = (snap_points[:, 0].reshape(-1, 1) - _xs.reshape(1, -1)) ** 2 + (
                        snap_points[:, 1].reshape(-1, 1) - _ys.reshape(1, -1)) ** 2
            _loss = np.sum(np.amin(distances2, axis=0)**0.5)  # near-field
            # _loss = np.sum(np.amin(distances2**0.5, axis=1))  # MAE
            # _loss = np.sum(np.amin(distances2, axis=1))  # MSE

            return _loss / num_elements

        results = minimize(fun=loss,
                           x0=initial_guess,
                           bounds=bounds,
                           method="L-BFGS-B")
                        # method="Powell")
                        # method="TNC")

        params = results.x

        _d_xs = np.cumsum(params[: num_elements - 1] * np.cos(
            params[2 * num_elements - 2] + params[num_elements - 1: 2 * num_elements - 2]))
        _d_ys = np.cumsum(params[: num_elements - 1] * np.sin(
            params[2 * num_elements - 2] + params[num_elements - 1: 2 * num_elements - 2]))
        _xs = np.hstack([params[2 * num_elements - 2 + d_ind_a + 0],
                         params[2 * num_elements - 2 + d_ind_a + 0] + _d_xs])
        _ys = np.hstack([params[2 * num_elements - 2 + d_ind_a + 1],
                         params[2 * num_elements - 2 + d_ind_a + 1] + _d_ys])

        # print("done fitting")
        return _xs, _ys, results.fun
    else:
        return motion_x, motion_y, -1


def optimize_trajectory_fs_v2(motion_x, motion_y, motion_t,
                              fs_points,
                              leaked_record,
                              snap_range,  # +-snap/start_point shift range in fraction of x/y bounding box (max-min)
                              step_range, angle_range,
                              angle0_range):  # +-step_range in fraction of step length; +-angle_range in fraction of Pi

    # REVERSE START <-> END POINTS?
    is_start_semivalid = leaked_record["start_x"] > 0 and leaked_record["start_delay"] > 0
    is_end_semivalid = leaked_record["end_x"] > 0 and leaked_record["end_delay"] > 0

    if is_start_semivalid and is_end_semivalid:
        is_reversed = leaked_record["start_delay"] > leaked_record["end_delay"]
    elif is_end_semivalid:
        is_reversed = True
    else:
        is_reversed = False

    is_reversed = True

    leaked_delay = leaked_record["end_delay"] if is_reversed else leaked_record["start_delay"]
    leaked_x = leaked_record["end_x"] if is_reversed else leaked_record["start_x"]
    leaked_y = leaked_record["end_y"] if is_reversed else leaked_record["start_y"]

    if is_reversed:
        motion_x = motion_x[::-1]
        motion_y = motion_y[::-1]
        motion_t = motion_t[::-1]
    ###################################################################
    num_elements = motion_t.shape[0]
    trajectory_extent = ((motion_x.max() - motion_x.min()) ** 2 + (motion_y.max() - motion_y.min()) ** 2) ** 0.5
    trajectory_extent_x = abs(motion_x.max() - motion_x.min())
    trajectory_extent_y = abs(motion_y.max() - motion_y.min())

    snap_distance = snap_range * trajectory_extent
    snap_distance_x = snap_range * trajectory_extent_x
    snap_distance_y = snap_range * trajectory_extent_y

    time_deltas = np.abs(np.diff(motion_t))  # size "n-1"
    motion_dx = np.diff(motion_x)
    motion_dy = np.diff(motion_y)

    motion_steps_lengths_input = (motion_dx ** 2 + motion_dy ** 2) ** 0.5
    motion_angles_input = np.arctan2(motion_dy, motion_dx)
    # print(num_elements)
    ################  INITIAL/BOUNDARIES VALUES   ##################################
    step_lengths_ini = np.amin(np.vstack([time_deltas * V_LIMIT, motion_steps_lengths_input]), axis=0)
    angles_ini = motion_angles_input

    step_lengths_min = np.amin(np.vstack([time_deltas * V_LIMIT * 0.99, motion_steps_lengths_input * (1 - step_range)]),
                               axis=0)
    step_lengths_max = np.amin(np.vstack([time_deltas * V_LIMIT * 1.00, motion_steps_lengths_input * (1 + step_range)]),
                               axis=0)  # not more than maximum speed nor allowed variation of step length
    angles_min = motion_angles_input - np.pi * angle_range
    angles_max = motion_angles_input + np.pi * angle_range

    angle0_ini = np.array([0.0])

    initial_start = np.array([motion_x[0], motion_y[0]])

    x_start_min = initial_start[0] - snap_distance_x
    x_start_max = initial_start[0] + snap_distance_x
    y_start_min = initial_start[1] - snap_distance_y
    y_start_max = initial_start[1] + snap_distance_y
    ############# Limit snap range by leaked data (if narrower and leaked data valid)  #######
    if leaked_x > 0 and leaked_delay > 0 and (V_LIMIT * leaked_delay) < trajectory_extent * INVALID_RANGE:
        _x_start_min = max(leaked_x - V_LIMIT * leaked_delay, x_start_min)
        _x_start_max = min(leaked_x + V_LIMIT * leaked_delay, x_start_max)
        _y_start_min = max(leaked_y - V_LIMIT * leaked_delay, y_start_min)
        _y_start_max = min(leaked_y + V_LIMIT * leaked_delay, y_start_max)

        x_start_min, x_start_max = (_x_start_min, _x_start_max) if _x_start_min < _x_start_max else (
        x_start_min, x_start_max)
        y_start_min, y_start_max = (_y_start_min, _y_start_max) if _y_start_min < _y_start_max else (
        y_start_min, y_start_max)

        #initial_start = np.array([(x_start_min + x_start_max) / 2, (y_start_min + y_start_max) / 2])
    ################ RESCALE XY0/INI-BOUNDS  ####################
    initial_start *= SCALE_XY0
    x_start_min *= SCALE_XY0
    x_start_max *= SCALE_XY0
    y_start_min *= SCALE_XY0
    y_start_max *= SCALE_XY0
    #################################################################
    bounds_angle0 = [(-angle0_range * np.pi, angle0_range * np.pi)]
    bounds_start = [(0, None), (0, None)]
    #bounds_start = [(x_start_min, x_start_max), (y_start_min, y_start_max)]  # 2x pairs
    bounds_steps = [(min_el, max_el) for min_el, max_el in zip(step_lengths_min, step_lengths_max)]  # (n-1)x pairs
    bounds_angles = [(min_el, max_el) for min_el, max_el in zip(angles_min, angles_max)]  # (n-1)x pairs

    initial_guess = np.concatenate([step_lengths_ini, angles_ini, angle0_ini, initial_start])  # (2*n+2)x
    bounds = bounds_steps + bounds_angles + bounds_angle0 + bounds_start  # (2*n+2)x pairs
    ###############################################################################
    #########################  SNAPPING POINTS   ##################################
    snap_points = []

    for i_step, _ in enumerate(motion_x):
        _distances = ((fs_points[:, 0] - motion_x[i_step]) ** 2 + (fs_points[:, 1] - motion_y[i_step]) ** 2) ** 0.5
        _snap_points = fs_points[_distances <= snap_distance]

        if len(_snap_points) > 0:
            snap_points.append(_snap_points)

    snap_points = np.concatenate(snap_points)
    snap_points = np.array(list(set(map(tuple, snap_points))))

    # print("snap_distance", snap_distance)
    # print("snap_points.shape", snap_points.shape)
    ##################################################################################
    ######################### OPTIMIZE ###############################################
    if len(snap_points) > 0:  # some points are within snapping range

        def loss(params):
            # print(params.shape)
            _d_xs = np.cumsum(params[: num_elements - 1] * np.cos(
                params[2 * num_elements - 2] + params[num_elements - 1: 2 * num_elements - 2]))
            _d_ys = np.cumsum(params[: num_elements - 1] * np.sin(
                params[2 * num_elements - 2] + params[num_elements - 1: 2 * num_elements - 2]))
            _xs = np.hstack([params[2 * num_elements - 1] / SCALE_XY0,
                             params[2 * num_elements - 1] / SCALE_XY0 + _d_xs])
            _ys = np.hstack([params[2 * num_elements] / SCALE_XY0,
                             params[2 * num_elements] / SCALE_XY0 + _d_ys])

            distances2 = (snap_points[:, 0].reshape(-1, 1) - _xs.reshape(1, -1)) ** 2 + (
                        snap_points[:, 1].reshape(-1, 1) - _ys.reshape(1, -1)) ** 2
            _loss = np.sum(np.amin(distances2, axis=0))  # near-field

            return _loss / num_elements

        results = minimize(fun=loss,
                           x0=initial_guess,
                           bounds=bounds,
                           options={'maxcor': 30, 'ftol': 1e-08, 'gtol': 1e-07, 'maxfun': 20000, 'maxiter': 20000,
                                    'maxls': 30},
                           method="L-BFGS-B")

        params = results.x

        _d_xs = np.cumsum(params[: num_elements - 1] * np.cos(
            params[2 * num_elements - 2] + params[num_elements - 1: 2 * num_elements - 2]))
        _d_ys = np.cumsum(params[: num_elements - 1] * np.sin(
            params[2 * num_elements - 2] + params[num_elements - 1: 2 * num_elements - 2]))
        _xs = np.hstack([params[2 * num_elements - 1] / SCALE_XY0,
                         params[2 * num_elements - 1] / SCALE_XY0 + _d_xs])
        _ys = np.hstack([params[2 * num_elements] / SCALE_XY0,
                         params[2 * num_elements] / SCALE_XY0 + _d_ys])

        if is_reversed:
            return _xs[::-1], _ys[::-1], results.fun, results.success, results.message
        else:
            return _xs, _ys, results.fun, results.success, results.message
    else:
        return motion_x, motion_y, -1, False, ""

def optimize_trajectory_fs_v2F(motion_x, motion_y, motion_t,
                              fs_points,
                              leaked_record,
                              snap_range,  # +-snap/start_point shift range in fraction of x/y bounding box (max-min)
                              step_range, angle_range,
                              angle0_range):  # +-step_range in fraction of step length; +-angle_range in fraction of Pi

    # REVERSE START <-> END POINTS?
    is_start_semivalid = leaked_record["start_x"] > 0 and leaked_record["start_delay"] > 0
    is_end_semivalid = leaked_record["end_x"] > 0 and leaked_record["end_delay"] > 0

    if is_start_semivalid and is_end_semivalid:
        is_reversed = leaked_record["start_delay"] > leaked_record["end_delay"]
    elif is_end_semivalid:
        is_reversed = True
    else:
        is_reversed = False

    is_reversed = False

    leaked_delay = leaked_record["end_delay"] if is_reversed else leaked_record["start_delay"]
    leaked_x = leaked_record["end_x"] if is_reversed else leaked_record["start_x"]
    leaked_y = leaked_record["end_y"] if is_reversed else leaked_record["start_y"]

    if is_reversed:
        motion_x = motion_x[::-1]
        motion_y = motion_y[::-1]
        motion_t = motion_t[::-1]
    ###################################################################
    num_elements = motion_t.shape[0]
    trajectory_extent = ((motion_x.max() - motion_x.min()) ** 2 + (motion_y.max() - motion_y.min()) ** 2) ** 0.5
    trajectory_extent_x = abs(motion_x.max() - motion_x.min())
    trajectory_extent_y = abs(motion_y.max() - motion_y.min())

    snap_distance = snap_range * trajectory_extent
    snap_distance_x = snap_range * trajectory_extent_x
    snap_distance_y = snap_range * trajectory_extent_y

    time_deltas = np.abs(np.diff(motion_t))  # size "n-1"
    motion_dx = np.diff(motion_x)
    motion_dy = np.diff(motion_y)

    motion_steps_lengths_input = (motion_dx ** 2 + motion_dy ** 2) ** 0.5
    motion_angles_input = np.arctan2(motion_dy, motion_dx)
    # print(num_elements)
    ################  INITIAL/BOUNDARIES VALUES   ##################################
    step_lengths_ini = np.amin(np.vstack([time_deltas * V_LIMIT, motion_steps_lengths_input]), axis=0)
    angles_ini = motion_angles_input

    step_lengths_min = np.amin(np.vstack([time_deltas * V_LIMIT * 0.99, motion_steps_lengths_input * (1 - step_range)]),
                               axis=0)
    step_lengths_max = np.amin(np.vstack([time_deltas * V_LIMIT * 1.00, motion_steps_lengths_input * (1 + step_range)]),
                               axis=0)  # not more than maximum speed nor allowed variation of step length
    angles_min = motion_angles_input - np.pi * angle_range
    angles_max = motion_angles_input + np.pi * angle_range

    angle0_ini = np.array([0.0])

    initial_start = np.array([motion_x[0], motion_y[0]])

    x_start_min = initial_start[0] - snap_distance_x
    x_start_max = initial_start[0] + snap_distance_x
    y_start_min = initial_start[1] - snap_distance_y
    y_start_max = initial_start[1] + snap_distance_y
    ############# Limit snap range by leaked data (if narrower and leaked data valid)  #######
    if leaked_x > 0 and leaked_delay > 0 and (V_LIMIT * leaked_delay) < trajectory_extent * INVALID_RANGE:
        _x_start_min = max(leaked_x - V_LIMIT * leaked_delay, x_start_min)
        _x_start_max = min(leaked_x + V_LIMIT * leaked_delay, x_start_max)
        _y_start_min = max(leaked_y - V_LIMIT * leaked_delay, y_start_min)
        _y_start_max = min(leaked_y + V_LIMIT * leaked_delay, y_start_max)

        x_start_min, x_start_max = (_x_start_min, _x_start_max) if _x_start_min < _x_start_max else (
        x_start_min, x_start_max)
        y_start_min, y_start_max = (_y_start_min, _y_start_max) if _y_start_min < _y_start_max else (
        y_start_min, y_start_max)

        #initial_start = np.array([(x_start_min + x_start_max) / 2, (y_start_min + y_start_max) / 2])
    ################ RESCALE XY0/INI-BOUNDS  ####################
    initial_start *= SCALE_XY0
    x_start_min *= SCALE_XY0
    x_start_max *= SCALE_XY0
    y_start_min *= SCALE_XY0
    y_start_max *= SCALE_XY0
    #################################################################
    bounds_angle0 = [(-angle0_range * np.pi, angle0_range * np.pi)]
    bounds_start = [(0, None), (0, None)]
    #bounds_start = [(x_start_min, x_start_max), (y_start_min, y_start_max)]  # 2x pairs
    bounds_steps = [(min_el, max_el) for min_el, max_el in zip(step_lengths_min, step_lengths_max)]  # (n-1)x pairs
    bounds_angles = [(min_el, max_el) for min_el, max_el in zip(angles_min, angles_max)]  # (n-1)x pairs

    initial_guess = np.concatenate([step_lengths_ini, angles_ini, angle0_ini, initial_start])  # (2*n+2)x
    bounds = bounds_steps + bounds_angles + bounds_angle0 + bounds_start  # (2*n+2)x pairs
    ###############################################################################
    #########################  SNAPPING POINTS   ##################################
    snap_points = []

    for i_step, _ in enumerate(motion_x):
        _distances = ((fs_points[:, 0] - motion_x[i_step]) ** 2 + (fs_points[:, 1] - motion_y[i_step]) ** 2) ** 0.5
        _snap_points = fs_points[_distances <= snap_distance]

        if len(_snap_points) > 0:
            snap_points.append(_snap_points)

    snap_points = np.concatenate(snap_points)
    snap_points = np.array(list(set(map(tuple, snap_points))))

    # print("snap_distance", snap_distance)
    # print("snap_points.shape", snap_points.shape)
    ##################################################################################
    ######################### OPTIMIZE ###############################################
    if len(snap_points) > 0:  # some points are within snapping range

        def loss(params):
            # print(params.shape)
            _d_xs = np.cumsum(params[: num_elements - 1] * np.cos(
                params[2 * num_elements - 2] + params[num_elements - 1: 2 * num_elements - 2]))
            _d_ys = np.cumsum(params[: num_elements - 1] * np.sin(
                params[2 * num_elements - 2] + params[num_elements - 1: 2 * num_elements - 2]))
            _xs = np.hstack([params[2 * num_elements - 1] / SCALE_XY0,
                             params[2 * num_elements - 1] / SCALE_XY0 + _d_xs])
            _ys = np.hstack([params[2 * num_elements] / SCALE_XY0,
                             params[2 * num_elements] / SCALE_XY0 + _d_ys])

            distances2 = (snap_points[:, 0].reshape(-1, 1) - _xs.reshape(1, -1)) ** 2 + (
                        snap_points[:, 1].reshape(-1, 1) - _ys.reshape(1, -1)) ** 2
            _loss = np.sum(np.amin(distances2, axis=0))  # near-field

            return _loss / num_elements

        results = minimize(fun=loss,
                           x0=initial_guess,
                           bounds=bounds,
                           options={'maxcor': 30, 'ftol': 1e-08, 'gtol': 1e-07, 'maxfun': 20000, 'maxiter': 20000,
                                    'maxls': 30},
                           method="L-BFGS-B")

        params = results.x

        _d_xs = np.cumsum(params[: num_elements - 1] * np.cos(
            params[2 * num_elements - 2] + params[num_elements - 1: 2 * num_elements - 2]))
        _d_ys = np.cumsum(params[: num_elements - 1] * np.sin(
            params[2 * num_elements - 2] + params[num_elements - 1: 2 * num_elements - 2]))
        _xs = np.hstack([params[2 * num_elements - 1] / SCALE_XY0,
                         params[2 * num_elements - 1] / SCALE_XY0 + _d_xs])
        _ys = np.hstack([params[2 * num_elements] / SCALE_XY0,
                         params[2 * num_elements] / SCALE_XY0 + _d_ys])

        if is_reversed:
            return _xs[::-1], _ys[::-1], results.fun, results.success, results.message
        else:
            return _xs, _ys, results.fun, results.success, results.message
    else:
        return motion_x, motion_y, -1, False, ""


def optimize_trajectory_fs_v3(motion_x, motion_y, motion_t,
                              fs_points,
                              leaked_record,
                              snap_range,  # +-snap/start_point shift range in fraction of x/y bounding box (max-min)
                              step_range, angle_range,
                              angle0_range):  # +-step_range in fraction of step length; +-angle_range in fraction of Pi

    # REVERSE START <-> END POINTS?
    is_start_semivalid = leaked_record["start_x"] > 0 and leaked_record["start_delay"] > 0
    is_end_semivalid = leaked_record["end_x"] > 0 and leaked_record["end_delay"] > 0

    if is_start_semivalid and is_end_semivalid:
        is_reversed = leaked_record["start_delay"] > leaked_record["end_delay"]
    elif is_end_semivalid:
        is_reversed = True
    else:
        is_reversed = False

    leaked_delay = leaked_record["end_delay"] if is_reversed else leaked_record["start_delay"]
    leaked_x = leaked_record["end_x"] if is_reversed else leaked_record["start_x"]
    leaked_y = leaked_record["end_y"] if is_reversed else leaked_record["start_y"]

    if is_reversed:
        motion_x = motion_x[::-1]
        motion_y = motion_y[::-1]
        motion_t = motion_t[::-1]
    ###################################################################
    num_elements = motion_t.shape[0]
    trajectory_extent = ((motion_x.max() - motion_x.min()) ** 2 + (motion_y.max() - motion_y.min()) ** 2) ** 0.5
    trajectory_extent_x = abs(motion_x.max() - motion_x.min())
    trajectory_extent_y = abs(motion_y.max() - motion_y.min())

    snap_distance = snap_range * trajectory_extent
    snap_distance_x = snap_range * trajectory_extent_x
    snap_distance_y = snap_range * trajectory_extent_y

    time_deltas = np.abs(np.diff(motion_t))  # size "n-1"
    motion_dx = np.diff(motion_x)
    motion_dy = np.diff(motion_y)

    motion_steps_lengths_input = (motion_dx ** 2 + motion_dy ** 2) ** 0.5
    motion_angles_input = np.arctan2(motion_dy, motion_dx)
    ################  INITIAL/BOUNDARIES VALUES   ##################################
    angle0_ini = np.array([motion_angles_input[0]])
    angle0_min = np.minimum(motion_angles_input[0] - angle0_range * np.pi,
                            motion_angles_input[0] + angle0_range * np.pi)
    angle0_max = np.maximum(motion_angles_input[0] - angle0_range * np.pi,
                            motion_angles_input[0] + angle0_range * np.pi)
    # DELTA OF ANGLE CHANGE (in abs units) (n-1) segments-1 => n-2
    angles_ini = np.diff(motion_angles_input)
    angles_min = np.minimum(angles_ini * (1 - angle_range), angles_ini * (1 + angle_range))
    angles_max = np.maximum(angles_ini * (1 - angle_range), angles_ini * (1 + angle_range))

    step_lengths_ini = np.minimum(time_deltas * V_LIMIT, motion_steps_lengths_input)
    step_lengths_min = np.minimum(time_deltas * V_LIMIT * 0.99, motion_steps_lengths_input * (1 - step_range))
    step_lengths_max = np.minimum(time_deltas * V_LIMIT * 1.00, motion_steps_lengths_input * (
                1 + step_range))  # not more than maximum speed nor allowed variation of step length

    initial_start = np.array([motion_x[0], motion_y[0]])
    x_start_min = initial_start[0] - snap_distance_x
    x_start_max = initial_start[0] + snap_distance_x
    y_start_min = initial_start[1] - snap_distance_y
    y_start_max = initial_start[1] + snap_distance_y
    ############# Limit snap range by leaked data (if narrower and leaked data valid)  #######
    if leaked_x > 0 and leaked_delay > 0 and (V_LIMIT * leaked_delay) < trajectory_extent * INVALID_RANGE:
        _x_start_min = max(leaked_x - V_LIMIT * leaked_delay, x_start_min)
        _x_start_max = min(leaked_x + V_LIMIT * leaked_delay, x_start_max)
        _y_start_min = max(leaked_y - V_LIMIT * leaked_delay, y_start_min)
        _y_start_max = min(leaked_y + V_LIMIT * leaked_delay, y_start_max)

        x_start_min, x_start_max = (_x_start_min, _x_start_max) if _x_start_min < _x_start_max else (
        x_start_min, x_start_max)
        y_start_min, y_start_max = (_y_start_min, _y_start_max) if _y_start_min < _y_start_max else (
        y_start_min, y_start_max)

        # initial_start = np.array([(x_start_min+x_start_max)/2, (y_start_min+y_start_max)/2])
    ################ RESCALE XY0/INI-BOUNDS  ####################
    initial_start *= SCALE_XY0
    x_start_min *= SCALE_XY0
    x_start_max *= SCALE_XY0
    y_start_min *= SCALE_XY0
    y_start_max *= SCALE_XY0
    #################################################################
    bounds_angle0 = [(angle0_min, angle0_max)]
    bounds_start = [(0, None), (0, None)]
    # bounds_start = [(x_start_min, x_start_max), (y_start_min, y_start_max)]  # 2x pairs
    bounds_angles = [(min_el, max_el) for min_el, max_el in zip(angles_min, angles_max)]  # (n-2)x pairs
    bounds_steps = [(min_el, max_el) for min_el, max_el in zip(step_lengths_min, step_lengths_max)]  # (n-1)x pairs

    initial_guess = np.concatenate([step_lengths_ini, angles_ini, angle0_ini, initial_start])  # (2*n+2)x
    bounds = bounds_steps + bounds_angles + bounds_angle0 + bounds_start  # (2*n+2)x pairs

    # print(initial_guess)
    # print(bounds)

    # :num_elements-1 => step_lengths (n-1)
    # num_elements-1:2*num_elements-3 => angle_changes (n-2)
    # 2*num_elements-3 => angle0 (1)
    # 2*num_elements-2 => x0 (1)
    # 2*num_elements-1 => y0 (1)

    ###############################################################################
    #########################  SNAPPING POINTS   ##################################
    snap_points = []

    for i_step, _ in enumerate(motion_x):
        _distances = ((fs_points[:, 0] - motion_x[i_step]) ** 2 + (fs_points[:, 1] - motion_y[i_step]) ** 2) ** 0.5
        _snap_points = fs_points[_distances <= snap_distance]

        if len(_snap_points) > 0:
            snap_points.append(_snap_points)

    snap_points = np.concatenate(snap_points)
    snap_points = np.array(list(set(map(tuple, snap_points))))

    # print("snap_distance", snap_distance)
    # print("snap_points.shape", snap_points.shape)
    ##################################################################################
    ######################### OPTIMIZE ###############################################
    if len(snap_points) > 0:  # some points are within snapping range

        def loss(params):
            # print(params.shape)
            _d_as = np.cumsum(params[num_elements - 1:2 * num_elements - 3])
            _as = np.hstack([params[2 * num_elements - 3],
                             params[2 * num_elements - 3] + _d_as])

            _d_xs = np.cumsum(params[:num_elements - 1] * np.cos(_as))
            _d_ys = np.cumsum(params[:num_elements - 1] * np.sin(_as))

            _xs = np.hstack([params[2 * num_elements - 2] / SCALE_XY0,
                             params[2 * num_elements - 2] / SCALE_XY0 + _d_xs])
            _ys = np.hstack([params[2 * num_elements - 1] / SCALE_XY0,
                             params[2 * num_elements - 1] / SCALE_XY0 + _d_ys])
            # print("_xs.shape", _xs.shape)
            distances2 = (snap_points[:, 0].reshape(-1, 1) - _xs.reshape(1, -1)) ** 2 + (
                        snap_points[:, 1].reshape(-1, 1) - _ys.reshape(1, -1)) ** 2
            _loss = np.sum(np.amin(distances2, axis=0))

            return _loss / num_elements

        results = minimize(fun=loss,
                           x0=initial_guess,
                           bounds=bounds,
                           options={'maxcor': 30, 'ftol': 1e-08, 'gtol': 1e-07, 'maxfun': 20000, 'maxiter': 20000,
                                    'maxls': 30},
                           method="L-BFGS-B")

        params = results.x

        _d_as = np.cumsum(params[num_elements - 1:2 * num_elements - 3])
        _as = np.hstack([params[2 * num_elements - 3],
                         params[2 * num_elements - 3] + _d_as])

        _d_xs = np.cumsum(params[:num_elements - 1] * np.cos(_as))
        _d_ys = np.cumsum(params[:num_elements - 1] * np.sin(_as))

        _xs = np.hstack([params[2 * num_elements - 2] / SCALE_XY0,
                         params[2 * num_elements - 2] / SCALE_XY0 + _d_xs])
        _ys = np.hstack([params[2 * num_elements - 1] / SCALE_XY0,
                         params[2 * num_elements - 1] / SCALE_XY0 + _d_ys])

        # print("done fitting")
        if is_reversed:
            return _xs[::-1], _ys[::-1], results.fun, results.success, results.message
        else:
            return _xs, _ys, results.fun, results.success, results.message
    else:
        return motion_x, motion_y, -1, False, ""

def optimize_trajectory_fs_v4(motion_x_full, motion_y_full, motion_t_full,
                              fs_points_full,
                              leaked_record,
                              snap_range,  # +-snap/start_point shift range in fraction of x/y bounding box (max-min)
                              step_range, angle_range,
                              angle0_range):  # +-step_range in fraction of step length; +-angle_range in fraction of Pi

    num_cuts = np.ceil((leaked_record["end_t"] - leaked_record["start_t"]) / T_CUT).astype(int)
    _t0 = motion_t_full[0]

    sum_xs = []
    sum_ys = []
    for i_cut in range(num_cuts):
        # prepare data for a given segement
        local_inds = motion_t_full[(_t0 + T_CUT * i_cut <= motion_t_full) & (motion_t_full < _t0 + T_CUT * (i_cut + 1))]

        motion_x = motion_x_full[local_inds]
        motion_y = motion_y_full[local_inds]
        motion_t = motion_t_full[local_inds]
        fs_points = fs_points_full[local_inds]
        ###################################################################
        num_elements = motion_t.shape[0]
        trajectory_extent = ((motion_x.max() - motion_x.min()) ** 2 + (motion_y.max() - motion_y.min()) ** 2) ** 0.5
        trajectory_extent_x = abs(motion_x.max() - motion_x.min())
        trajectory_extent_y = abs(motion_y.max() - motion_y.min())

        snap_distance = snap_range * trajectory_extent
        snap_distance_x = snap_range * trajectory_extent_x
        snap_distance_y = snap_range * trajectory_extent_y

        time_deltas = np.abs(np.diff(motion_t))  # size "n-1"
        motion_dx = np.diff(motion_x)
        motion_dy = np.diff(motion_y)

        motion_steps_lengths_input = (motion_dx ** 2 + motion_dy ** 2) ** 0.5
        motion_angles_input = np.arctan2(motion_dy, motion_dx)
        # print(num_elements)
        ################  INITIAL/BOUNDARIES VALUES   ##################################
        step_lengths_ini = np.amin(np.vstack([time_deltas * V_LIMIT, motion_steps_lengths_input]), axis=0)
        angles_ini = motion_angles_input

        step_lengths_min = np.amin(
            np.vstack([time_deltas * V_LIMIT * 0.99, motion_steps_lengths_input * (1 - step_range)]),
            axis=0)
        step_lengths_max = np.amin(
            np.vstack([time_deltas * V_LIMIT * 1.00, motion_steps_lengths_input * (1 + step_range)]),
            axis=0)  # not more than maximum speed nor allowed variation of step length
        angles_min = motion_angles_input - np.pi * angle_range
        angles_max = motion_angles_input + np.pi * angle_range

        angle0_ini = np.array([0.0])

        initial_start = np.array([motion_x[0], motion_y[0]])

        x_start_min = initial_start[0] - snap_distance_x
        x_start_max = initial_start[0] + snap_distance_x
        y_start_min = initial_start[1] - snap_distance_y
        y_start_max = initial_start[1] + snap_distance_y

        ################ RESCALE XY0/INI-BOUNDS  ####################
        initial_start *= SCALE_XY0
        x_start_min *= SCALE_XY0
        x_start_max *= SCALE_XY0
        y_start_min *= SCALE_XY0
        y_start_max *= SCALE_XY0
        #################################################################
        bounds_angle0 = [(-angle0_range * np.pi, angle0_range * np.pi)]
        bounds_start = [(0, None), (0, None)]
        # bounds_start = [(x_start_min, x_start_max), (y_start_min, y_start_max)]  # 2x pairs
        bounds_steps = [(min_el, max_el) for min_el, max_el in zip(step_lengths_min, step_lengths_max)]  # (n-1)x pairs
        bounds_angles = [(min_el, max_el) for min_el, max_el in zip(angles_min, angles_max)]  # (n-1)x pairs

        initial_guess = np.concatenate([step_lengths_ini, angles_ini, angle0_ini, initial_start])  # (2*n+2)x
        bounds = bounds_steps + bounds_angles + bounds_angle0 + bounds_start  # (2*n+2)x pairs
        ###############################################################################
        #########################  SNAPPING POINTS   ##################################
        snap_points = []

        for i_step, _ in enumerate(motion_x):
            _distances = ((fs_points[:, 0] - motion_x[i_step]) ** 2 + (fs_points[:, 1] - motion_y[i_step]) ** 2) ** 0.5
            _snap_points = fs_points[_distances <= snap_distance]

            if len(_snap_points) > 0:
                snap_points.append(_snap_points)

        snap_points = np.concatenate(snap_points)
        snap_points = np.array(list(set(map(tuple, snap_points))))

        # print("snap_distance", snap_distance)
        # print("snap_points.shape", snap_points.shape)
        ##################################################################################
        ######################### OPTIMIZE ###############################################
        if len(snap_points) > 0:  # some points are within snapping range

            def loss(params):
                # print(params.shape)
                _d_xs = np.cumsum(params[: num_elements - 1] * np.cos(
                    params[2 * num_elements - 2] + params[num_elements - 1: 2 * num_elements - 2]))
                _d_ys = np.cumsum(params[: num_elements - 1] * np.sin(
                    params[2 * num_elements - 2] + params[num_elements - 1: 2 * num_elements - 2]))
                _xs = np.hstack([params[2 * num_elements - 1] / SCALE_XY0,
                                 params[2 * num_elements - 1] / SCALE_XY0 + _d_xs])
                _ys = np.hstack([params[2 * num_elements] / SCALE_XY0,
                                 params[2 * num_elements] / SCALE_XY0 + _d_ys])

                distances2 = (snap_points[:, 0].reshape(-1, 1) - _xs.reshape(1, -1)) ** 2 + (
                        snap_points[:, 1].reshape(-1, 1) - _ys.reshape(1, -1)) ** 2
                _loss = np.sum(np.amin(distances2, axis=0))  # near-field

                return _loss / num_elements

            results = minimize(fun=loss,
                               x0=initial_guess,
                               bounds=bounds,
                               options={'maxcor': 30, 'ftol': 1e-08, 'gtol': 1e-07, 'maxfun': 20000, 'maxiter': 20000,
                                        'maxls': 30},
                               method="L-BFGS-B")

            params = results.x

            _d_xs = np.cumsum(params[: num_elements - 1] * np.cos(
                params[2 * num_elements - 2] + params[num_elements - 1: 2 * num_elements - 2]))
            _d_ys = np.cumsum(params[: num_elements - 1] * np.sin(
                params[2 * num_elements - 2] + params[num_elements - 1: 2 * num_elements - 2]))
            _xs = np.hstack([params[2 * num_elements - 1] / SCALE_XY0,
                             params[2 * num_elements - 1] / SCALE_XY0 + _d_xs])
            _ys = np.hstack([params[2 * num_elements] / SCALE_XY0,
                             params[2 * num_elements] / SCALE_XY0 + _d_ys])

            sum_xs.append(_xs)
            sum_ys.append(_ys)
        else:
            sum_xs.append(motion_x)
            sum_ys.append(motion_y)

        return np.concatenate(sum_xs), np.concatenate(sum_ys), -1, False, ""

def motion_multi(site_id, coarse_data, motion_data, leaked_data, step_range, angle_range, angle0_range):
    print("---------------------------------------------------")
    print(f"Starting {site_id} at {time.ctime(time.time())}")

    snapped_data_site_id = {}
    for trace_id in coarse_data: #tqdm(coarse_data):  # over traces
        # print(trace_id)
        coarse_record = coarse_data[trace_id][["time", "x", "y"]]
        motion_record = motion_data[trace_id][["time", "rx", "ry"]]
        leaked_record = leaked_data[trace_id]

        _x, _y, _fun = optimize_trajectory_motion(motion_record.rx.to_numpy(),
                                                  motion_record.ry.to_numpy(),
                                                  motion_record.time.to_numpy(),  # sensor data (relative steps vs cumulative)
                                                  coarse_record.x.to_numpy(),
                                                  coarse_record.y.to_numpy(),
                                                  coarse_record.time.to_numpy(),  # wifi data
                                                  leaked_record,
                                                  step_range, angle_range, angle0_range)

        snapped_data_site_id[trace_id] = pd.DataFrame(_x, columns=["x"])
        snapped_data_site_id[trace_id]["y"] = _y
        snapped_data_site_id[trace_id]["floor"] = leaked_record["floor"]
        snapped_data_site_id[trace_id]["time"] = motion_record.time.to_numpy()

    print(f"Done with {site_id} at {time.ctime(time.time())}")
    print("---------------------------------------------------")

    return {site_id: snapped_data_site_id}


def fs_multi(site_id, predicted_data, fs_data, leaked_data,
             snap_range, step_range, angle_range, angle0_range):
    print("---------------------------------------------------")
    print(f"Starting {site_id} at {time.ctime(time.time())}")

    isV1 = False

    snapped_data_site_id = {}
    for trace_id in predicted_data:  # over traces
        # print(trace_id)
        predicted_record = predicted_data[trace_id][["time", "x", "y"]]
        _floor = predicted_data[trace_id]["floor"][0]
        fs_record = fs_data[_floor]
        leaked_record = leaked_data[trace_id]

        if isV1:
            _x, _y, _fun = optimize_trajectory_fs_v1(predicted_record.x.to_numpy(),
                                                     predicted_record.y.to_numpy(),
                                                     predicted_record.time.to_numpy(),
                                                     fs_record,
                                                     leaked_record,
                                                     snap_range, step_range, angle_range, angle0_range)
        else:
            _x, _y, _fun, _isOk, _exit_msg = optimize_trajectory_fs_v2(predicted_record.x.to_numpy(),
                                                                       predicted_record.y.to_numpy(),
                                                                       predicted_record.time.to_numpy(),
                                                                       fs_record,
                                                                       leaked_record,
                                                                       snap_range, step_range, angle_range, angle0_range)

        snapped_data_site_id[trace_id] = pd.DataFrame(_x, columns=["x"])
        snapped_data_site_id[trace_id]["y"] = _y
        snapped_data_site_id[trace_id]["floor"] = _floor
        snapped_data_site_id[trace_id]["time"] = predicted_record.time.to_numpy()

    print(f"Done with {site_id} at {time.ctime(time.time())}")
    print("---------------------------------------------------")
    return {site_id: snapped_data_site_id}

def fs_multiF(site_id, predicted_data, fs_data, leaked_data,
             snap_range, step_range, angle_range, angle0_range):
    print("---------------------------------------------------")
    print(f"Starting {site_id} at {time.ctime(time.time())}")

    isV1 = False

    snapped_data_site_id = {}
    for trace_id in predicted_data:  # over traces
        # print(trace_id)
        predicted_record = predicted_data[trace_id][["time", "x", "y"]]
        _floor = predicted_data[trace_id]["floor"][0]
        fs_record = fs_data[_floor]
        leaked_record = leaked_data[trace_id]

        if isV1:
            _x, _y, _fun = optimize_trajectory_fs_v1(predicted_record.x.to_numpy(),
                                                     predicted_record.y.to_numpy(),
                                                     predicted_record.time.to_numpy(),
                                                     fs_record,
                                                     leaked_record,
                                                     snap_range, step_range, angle_range, angle0_range)
        else:
            _x, _y, _fun, _isOk, _exit_msg = optimize_trajectory_fs_v2F(predicted_record.x.to_numpy(),
                                                                       predicted_record.y.to_numpy(),
                                                                       predicted_record.time.to_numpy(),
                                                                       fs_record,
                                                                       leaked_record,
                                                                       snap_range, step_range, angle_range, angle0_range)

        snapped_data_site_id[trace_id] = pd.DataFrame(_x, columns=["x"])
        snapped_data_site_id[trace_id]["y"] = _y
        snapped_data_site_id[trace_id]["floor"] = _floor
        snapped_data_site_id[trace_id]["time"] = predicted_record.time.to_numpy()

    print(f"Done with {site_id} at {time.ctime(time.time())}")
    print("---------------------------------------------------")
    return {site_id: snapped_data_site_id}


