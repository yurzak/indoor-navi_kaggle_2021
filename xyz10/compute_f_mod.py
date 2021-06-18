import numpy as np
import pandas as pd
import scipy.signal as signal

import copy
import random
import matplotlib
import matplotlib.pyplot as plt

def split_ts_seq(ts_seq, sep_ts):
    """

    :param ts_seq:
    :param sep_ts:
    :return:
    """
    tss = ts_seq[:, 0].astype(float)
    unique_sep_ts = np.unique(sep_ts)
    ts_seqs = []
    start_index = 0
    for i in range(0, unique_sep_ts.shape[0]):
        end_index = np.searchsorted(tss, unique_sep_ts[i], side='right')
        if start_index == end_index:
            continue
        ts_seqs.append(ts_seq[start_index:end_index, :].copy())
        start_index = end_index

    # tail data
    if start_index < ts_seq.shape[0]:
        ts_seqs.append(ts_seq[start_index:, :].copy())

    return ts_seqs

def split_ts_seq_mod(ts_seq, sep_ts):
    """

    :param ts_seq:
    :param sep_ts:
    :return:
    """
    ts_seq = ts_seq[(ts_seq[:, 0] >= sep_ts[0]) & (ts_seq[:, 0] <= sep_ts[-1])]  # only times between waypoint times
    tss = ts_seq[:, 0].astype(float)

    unique_sep_ts = np.unique(sep_ts)
    ts_seqs = []
    start_index = 0
    for i in range(0, unique_sep_ts.shape[0]):
        end_index = np.searchsorted(tss, unique_sep_ts[i], side='right')
        if start_index == end_index == 0:  # if first segment
            continue
        elif start_index == end_index != 0:  # if no relative steps between waypoint timestampts=> add zero step
            el = np.array([unique_sep_ts[i], 1e-4, 1e-4]).reshape(1, -1)
            ts_seqs.append(el)
        else:
            ts_seqs.append(ts_seq[start_index:end_index, :].copy())
        start_index = end_index

    # tail data
    if start_index < ts_seq.shape[0]:
        ts_seqs.append(ts_seq[start_index:, :].copy())

    return ts_seqs


def correct_trajectory(original_xys, end_xy):
    """

    :param original_xys: numpy ndarray, shape(N, 2)
    :param end_xy: numpy ndarray, shape(1, 2)
    :return:
    """
    corrected_xys = np.zeros((0, 2))

    A = original_xys[0, :]
    B = end_xy
    Bp = original_xys[-1, :]

    angle_BAX = np.arctan2(B[1] - A[1], B[0] - A[0])
    angle_BpAX = np.arctan2(Bp[1] - A[1], Bp[0] - A[0])
    angle_BpAB = angle_BpAX - angle_BAX
    AB = np.sqrt(np.sum((B - A) ** 2))
    ABp = np.sqrt(np.sum((Bp - A) ** 2))

    corrected_xys = np.append(corrected_xys, [A], 0)
    for i in np.arange(1, np.size(original_xys, 0)):
        angle_CpAX = np.arctan2(original_xys[i, 1] - A[1], original_xys[i, 0] - A[0])

        angle_CAX = angle_CpAX - angle_BpAB

        ACp = np.sqrt(np.sum((original_xys[i, :] - A) ** 2))

        AC = ACp * AB / ABp

        delta_C = np.array([AC * np.cos(angle_CAX), AC * np.sin(angle_CAX)])

        C = delta_C + A

        corrected_xys = np.append(corrected_xys, [C], 0)

    return corrected_xys


def correct_positions(rel_positions, reference_positions):
    """

    :param rel_positions:
    :param reference_positions:
    :return:
    """
    rel_positions_list = split_ts_seq(rel_positions, reference_positions[:, 0])
    if len(rel_positions_list) != reference_positions.shape[0] - 1:
        # print(f'Rel positions list size: {len(rel_positions_list)}, ref positions size: {reference_positions.shape[0]}')
        del rel_positions_list[-1]
    assert len(rel_positions_list) == reference_positions.shape[0] - 1

    corrected_positions = np.zeros((0, 3))
    for i, rel_ps in enumerate(rel_positions_list):
        start_position = reference_positions[i]
        end_position = reference_positions[i + 1]
        abs_ps = np.zeros(rel_ps.shape)
        abs_ps[:, 0] = rel_ps[:, 0]
        # abs_ps[:, 1:3] = rel_ps[:, 1:3] + start_position[1:3]
        abs_ps[0, 1:3] = rel_ps[0, 1:3] + start_position[1:3]
        for j in range(1, rel_ps.shape[0]):
            abs_ps[j, 1:3] = abs_ps[j-1, 1:3] + rel_ps[j, 1:3]
        abs_ps = np.insert(abs_ps, 0, start_position, axis=0)
        corrected_xys = correct_trajectory(abs_ps[:, 1:3], end_position[1:3])
        corrected_ps = np.column_stack((abs_ps[:, 0], corrected_xys))
        if i == 0:
            corrected_positions = np.append(corrected_positions, corrected_ps, axis=0)
        else:
            corrected_positions = np.append(corrected_positions, corrected_ps[1:], axis=0)

    corrected_positions = np.array(corrected_positions)

    return corrected_positions

def correct_positions_mod(rel_positions, reference_positions):
    """

    :param rel_positions:
    :param reference_positions:
    :return:
    """

    rel_positions_list = split_ts_seq(rel_positions, reference_positions[:, 0])
    rel_positions_list_extra = copy.deepcopy(rel_positions_list)

    times_ref = reference_positions[:, 0]
    len_refs = reference_positions.shape[0]

    for i_s, ref_t in enumerate(times_ref):
        if i_s == 0 or i_s == len_refs - 1:  # skip first end last element
            continue
        i_e = i_s + 1

        num_sequence_in = 0
        for seq_i, seq_t in enumerate(rel_positions_list_extra):
            if np.any((times_ref[i_s] < seq_t[:, 0]) & (seq_t[:, 0] <= times_ref[i_e])):
                num_sequence_in += 1

        if num_sequence_in < 1:
            time_ins = 0.5*(times_ref[i_s] + times_ref[i_e])

            rel_positions_list_extra.insert(i_s, np.array([[time_ins, 0.0001, 0.0001]]))  # insert single-row array

    if len(rel_positions_list_extra) != reference_positions.shape[0] - 1:
        del rel_positions_list_extra[-1]
    if len(rel_positions_list_extra) != reference_positions.shape[0] - 1:
        print(rel_positions_list_extra[:][:, 0])
        print("--------------------------")
        print(reference_positions[:][:, 0])
        print("--------------------------")
    assert len(rel_positions_list_extra) == reference_positions.shape[0] - 1

    corrected_positions = np.zeros((0, 3))
    for i, rel_ps in enumerate(rel_positions_list_extra):
        start_position = reference_positions[i]
        end_position = reference_positions[i + 1]
        abs_ps = np.zeros(rel_ps.shape)
        abs_ps[:, 0] = rel_ps[:, 0]
        # abs_ps[:, 1:3] = rel_ps[:, 1:3] + start_position[1:3]
        abs_ps[0, 1:3] = rel_ps[0, 1:3] + start_position[1:3]
        for j in range(1, rel_ps.shape[0]):
            abs_ps[j, 1:3] = abs_ps[j-1, 1:3] + rel_ps[j, 1:3]
        abs_ps = np.insert(abs_ps, 0, start_position, axis=0)
        corrected_xys = correct_trajectory(abs_ps[:, 1:3], end_position[1:3])
        corrected_ps = np.column_stack((abs_ps[:, 0], corrected_xys))
        if i == 0:
            corrected_positions = np.append(corrected_positions, corrected_ps, axis=0)
        else:
            corrected_positions = np.append(corrected_positions, corrected_ps[1:], axis=0)

    corrected_positions = np.array(corrected_positions)


    return corrected_positions

def correct_positions_mod2(rel_positions, reference_positions):
    """

    :param rel_positions:
    :param reference_positions:
    :return:
    """
    rel_positions_list = split_ts_seq_mod(rel_positions, reference_positions[:, 0])

    assert len(rel_positions_list) == reference_positions.shape[0] - 1

    corrected_positions = np.zeros((0, 3))
    for i, rel_ps in enumerate(rel_positions_list):
        start_position = reference_positions[i]
        end_position = reference_positions[i + 1]
        abs_ps = np.zeros(rel_ps.shape)
        abs_ps[:, 0] = rel_ps[:, 0]
        # abs_ps[:, 1:3] = rel_ps[:, 1:3] + start_position[1:3]
        abs_ps[0, 1:3] = rel_ps[0, 1:3] + start_position[1:3]
        for j in range(1, rel_ps.shape[0]):
            abs_ps[j, 1:3] = abs_ps[j-1, 1:3] + rel_ps[j, 1:3]
        abs_ps = np.insert(abs_ps, 0, start_position, axis=0)
        corrected_xys = correct_trajectory(abs_ps[:, 1:3], end_position[1:3])
        corrected_ps = np.column_stack((abs_ps[:, 0], corrected_xys))
        if i == 0:
            corrected_positions = np.append(corrected_positions, corrected_ps, axis=0)
        else:
            corrected_positions = np.append(corrected_positions, corrected_ps[1:], axis=0)

    corrected_positions = np.array(corrected_positions)

    return corrected_positions


def init_parameters_filter(sample_freq, warmup_data, cut_off_freq=2):
    order = 4
    filter_b, filter_a = signal.butter(order, cut_off_freq / (sample_freq / 2), 'low', False)
    zf = signal.lfilter_zi(filter_b, filter_a)
    _, zf = signal.lfilter(filter_b, filter_a, warmup_data, zi=zf)
    _, filter_zf = signal.lfilter(filter_b, filter_a, warmup_data, zi=zf)

    return filter_b, filter_a, filter_zf


def get_rotation_matrix_from_vector(rotation_vector):
    q1 = rotation_vector[0]
    q2 = rotation_vector[1]
    q3 = rotation_vector[2]

    if rotation_vector.size >= 4:
        q0 = rotation_vector[3]
    else:
        q0 = 1 - q1*q1 - q2*q2 - q3*q3
        if q0 > 0:
            q0 = np.sqrt(q0)
        else:
            q0 = 0

    sq_q1 = 2 * q1 * q1
    sq_q2 = 2 * q2 * q2
    sq_q3 = 2 * q3 * q3
    q1_q2 = 2 * q1 * q2
    q3_q0 = 2 * q3 * q0
    q1_q3 = 2 * q1 * q3
    q2_q0 = 2 * q2 * q0
    q2_q3 = 2 * q2 * q3
    q1_q0 = 2 * q1 * q0

    R = np.zeros((9,))
    if R.size == 9:
        R[0] = 1 - sq_q2 - sq_q3
        R[1] = q1_q2 - q3_q0
        R[2] = q1_q3 + q2_q0

        R[3] = q1_q2 + q3_q0
        R[4] = 1 - sq_q1 - sq_q3
        R[5] = q2_q3 - q1_q0

        R[6] = q1_q3 - q2_q0
        R[7] = q2_q3 + q1_q0
        R[8] = 1 - sq_q1 - sq_q2

        R = np.reshape(R, (3, 3))
    elif R.size == 16:
        R[0] = 1 - sq_q2 - sq_q3
        R[1] = q1_q2 - q3_q0
        R[2] = q1_q3 + q2_q0
        R[3] = 0.0

        R[4] = q1_q2 + q3_q0
        R[5] = 1 - sq_q1 - sq_q3
        R[6] = q2_q3 - q1_q0
        R[7] = 0.0

        R[8] = q1_q3 - q2_q0
        R[9] = q2_q3 + q1_q0
        R[10] = 1 - sq_q1 - sq_q2
        R[11] = 0.0

        R[12] = R[13] = R[14] = 0.0
        R[15] = 1.0

        R = np.reshape(R, (4, 4))

    return R


def get_orientation(R):
    flat_R = R.flatten()
    values = np.zeros((3,))
    if np.size(flat_R) == 9:
        values[0] = np.arctan2(flat_R[1], flat_R[4])
        values[1] = np.arcsin(-flat_R[7])
        values[2] = np.arctan2(-flat_R[6], flat_R[8])
    else:
        values[0] = np.arctan2(flat_R[1], flat_R[5])
        values[1] = np.arcsin(-flat_R[9])
        values[2] = np.arctan2(-flat_R[8], flat_R[10])

    return values


def compute_steps(acce_datas):
    step_timestamps = np.array([])
    step_indexs = np.array([], dtype=int)
    step_acce_max_mins = np.zeros((0, 4))
    sample_freq = 50
    window_size = 22
    low_acce_mag = 0.6
    step_criterion = 1
    interval_threshold = 250

    acce_max = np.zeros((2,))
    acce_min = np.zeros((2,))
    acce_binarys = np.zeros((window_size,), dtype=int)
    acce_mag_pre = 0
    state_flag = 0

    warmup_data = np.ones((window_size,)) * 9.81
    filter_b, filter_a, filter_zf = init_parameters_filter(sample_freq, warmup_data)
    acce_mag_window = np.zeros((window_size, 1))

    # detect steps according to acceleration magnitudes
    for i in np.arange(0, np.size(acce_datas, 0)):
        acce_data = acce_datas[i, :]
        acce_mag = np.sqrt(np.sum(acce_data[1:] ** 2))

        acce_mag_filt, filter_zf = signal.lfilter(filter_b, filter_a, [acce_mag], zi=filter_zf)
        acce_mag_filt = acce_mag_filt[0]

        acce_mag_window = np.append(acce_mag_window, [acce_mag_filt])
        acce_mag_window = np.delete(acce_mag_window, 0)
        mean_gravity = np.mean(acce_mag_window)
        acce_std = np.std(acce_mag_window)
        mag_threshold = np.max([low_acce_mag, 0.4 * acce_std])

        # detect valid peak or valley of acceleration magnitudes
        acce_mag_filt_detrend = acce_mag_filt - mean_gravity
        if acce_mag_filt_detrend > np.max([acce_mag_pre, mag_threshold]):
            # peak
            acce_binarys = np.append(acce_binarys, [1])
            acce_binarys = np.delete(acce_binarys, 0)
        elif acce_mag_filt_detrend < np.min([acce_mag_pre, -mag_threshold]):
            # valley
            acce_binarys = np.append(acce_binarys, [-1])
            acce_binarys = np.delete(acce_binarys, 0)
        else:
            # between peak and valley
            acce_binarys = np.append(acce_binarys, [0])
            acce_binarys = np.delete(acce_binarys, 0)

        if (acce_binarys[-1] == 0) and (acce_binarys[-2] == 1):
            if state_flag == 0:
                acce_max[:] = acce_data[0], acce_mag_filt
                state_flag = 1
            elif (state_flag == 1) and ((acce_data[0] - acce_max[0]) <= interval_threshold) and (
                    acce_mag_filt > acce_max[1]):
                acce_max[:] = acce_data[0], acce_mag_filt
            elif (state_flag == 2) and ((acce_data[0] - acce_max[0]) > interval_threshold):
                acce_max[:] = acce_data[0], acce_mag_filt
                state_flag = 1

        # choose reasonable step criterion and check if there is a valid step
        # save step acceleration data: step_acce_max_mins = [timestamp, max, min, variance]
        step_flag = False
        if step_criterion == 2:
            if (acce_binarys[-1] == -1) and ((acce_binarys[-2] == 1) or (acce_binarys[-2] == 0)):
                step_flag = True
        elif step_criterion == 3:
            if (acce_binarys[-1] == -1) and (acce_binarys[-2] == 0) and (np.sum(acce_binarys[:-2]) > 1):
                step_flag = True
        else:
            if (acce_binarys[-1] == 0) and acce_binarys[-2] == -1:
                if (state_flag == 1) and ((acce_data[0] - acce_min[0]) > interval_threshold):
                    acce_min[:] = acce_data[0], acce_mag_filt
                    state_flag = 2
                    step_flag = True
                elif (state_flag == 2) and ((acce_data[0] - acce_min[0]) <= interval_threshold) and (
                        acce_mag_filt < acce_min[1]):
                    acce_min[:] = acce_data[0], acce_mag_filt
        if step_flag:
            step_timestamps = np.append(step_timestamps, acce_data[0])
            step_indexs = np.append(step_indexs, [i])
            step_acce_max_mins = np.append(step_acce_max_mins,
                                           [[acce_data[0], acce_max[1], acce_min[1], acce_std ** 2]], axis=0)
        acce_mag_pre = acce_mag_filt_detrend

    return step_timestamps, step_indexs, step_acce_max_mins


def compute_stride_length(step_acce_max_mins):
    K = 0.4
    K_max = 0.8
    K_min = 0.4
    para_a0 = 0.21468084
    para_a1 = 0.09154517
    para_a2 = 0.02301998

    stride_lengths = np.zeros((step_acce_max_mins.shape[0], 2))
    k_real = np.zeros((step_acce_max_mins.shape[0], 2))
    step_timeperiod = np.zeros((step_acce_max_mins.shape[0] - 1, ))
    stride_lengths[:, 0] = step_acce_max_mins[:, 0]
    window_size = 2
    step_timeperiod_temp = np.zeros((0, ))

    # calculate every step period - step_timeperiod unit: second
    for i in range(0, step_timeperiod.shape[0]):
        step_timeperiod_data = (step_acce_max_mins[i + 1, 0] - step_acce_max_mins[i, 0]) / 1000
        step_timeperiod_temp = np.append(step_timeperiod_temp, [step_timeperiod_data])
        if step_timeperiod_temp.shape[0] > window_size:
            step_timeperiod_temp = np.delete(step_timeperiod_temp, [0])
        step_timeperiod[i] = np.sum(step_timeperiod_temp) / step_timeperiod_temp.shape[0]

    # calculate parameters by step period and acceleration magnitude variance
    k_real[:, 0] = step_acce_max_mins[:, 0]
    k_real[0, 1] = K
    for i in range(0, step_timeperiod.shape[0]):
        k_real[i + 1, 1] = np.max([(para_a0 + para_a1 / step_timeperiod[i] + para_a2 * step_acce_max_mins[i, 3]), K_min])
        k_real[i + 1, 1] = np.min([k_real[i + 1, 1], K_max]) * (K / K_min)

    # calculate every stride length by parameters and max and min data of acceleration magnitude
    stride_lengths[:, 1] = np.max([(step_acce_max_mins[:, 1] - step_acce_max_mins[:, 2]),
                                   np.ones((step_acce_max_mins.shape[0], ))], axis=0)**(1 / 4) * k_real[:, 1]

    return stride_lengths


def compute_headings(ahrs_datas):
    headings = np.zeros((np.size(ahrs_datas, 0), 2))
    for i in np.arange(0, np.size(ahrs_datas, 0)):
        ahrs_data = ahrs_datas[i, :]
        rot_mat = get_rotation_matrix_from_vector(ahrs_data[1:])
        azimuth, pitch, roll = get_orientation(rot_mat)
        around_z = (-azimuth) % (2 * np.pi)
        headings[i, :] = ahrs_data[0], around_z
    return headings


def compute_step_heading(step_timestamps, headings):
    step_headings = np.zeros((len(step_timestamps), 2))
    step_timestamps_index = 0
    for i in range(0, len(headings)):
        if step_timestamps_index < len(step_timestamps):
            if headings[i, 0] == step_timestamps[step_timestamps_index]:
                step_headings[step_timestamps_index, :] = headings[i, :]
                step_timestamps_index += 1
        else:
            break
    assert step_timestamps_index == len(step_timestamps)

    return step_headings


def compute_rel_positions(stride_lengths, step_headings):
    rel_positions = np.zeros((stride_lengths.shape[0], 3))
    for i in range(0, stride_lengths.shape[0]):
        rel_positions[i, 0] = stride_lengths[i, 0]
        rel_positions[i, 1] = -stride_lengths[i, 1] * np.sin(step_headings[i, 1])
        rel_positions[i, 2] = stride_lengths[i, 1] * np.cos(step_headings[i, 1])

    return rel_positions


def compute_step_positions(acce_datas, ahrs_datas, posi_datas):
    step_timestamps, step_indexs, step_acce_max_mins = compute_steps(acce_datas)
    headings = compute_headings(ahrs_datas)
    stride_lengths = compute_stride_length(step_acce_max_mins)
    step_headings = compute_step_heading(step_timestamps, headings)
    rel_positions = compute_rel_positions(stride_lengths, step_headings)
    step_positions = correct_positions(rel_positions, posi_datas)

    return step_positions

def compute_step_positions_mod(acce_datas, ahrs_datas, posi_datas):
    step_timestamps, step_indexs, step_acce_max_mins = compute_steps(acce_datas)
    headings = compute_headings(ahrs_datas)
    stride_lengths = compute_stride_length(step_acce_max_mins)
    step_headings = compute_step_heading(step_timestamps, headings)
    rel_positions = compute_rel_positions(stride_lengths, step_headings)
    step_positions = correct_positions_mod(rel_positions, posi_datas)

    return step_positions

def compute_step_positions_mod2(acce_datas, ahrs_datas, posi_datas):
    step_timestamps, step_indexs, step_acce_max_mins = compute_steps(acce_datas)
    headings = compute_headings(ahrs_datas)
    stride_lengths = compute_stride_length(step_acce_max_mins)
    step_headings = compute_step_heading(step_timestamps, headings)
    rel_positions = compute_rel_positions(stride_lengths, step_headings)
    step_positions = correct_positions_mod2(rel_positions, posi_datas)

    return step_positions

if __name__ == "__main__":

    rel_positions = np.array([[ 1.23900000e+03, -3.67817886e-01,  1.61523654e-01],
       [ 1.89700000e+03, -4.76079826e-01,  2.57843486e-01],
       [ 2.49400000e+03, -5.13848290e-01,  1.64554454e-01],
       [ 3.13200000e+03, -5.14406649e-01,  1.38747876e-01],
       [ 3.76900000e+03, -4.69338010e-01,  4.51000640e-02],
       [ 4.38700000e+03, -4.39625803e-01,  4.50117244e-02],
       [ 8.09300000e+03,  3.98057382e-01, -2.79919734e-01],
       [ 8.73000000e+03,  4.20787211e-01, -3.66589744e-01],
       [ 9.28800000e+03,  4.66479094e-01, -2.63749459e-01],
       [ 9.86600000e+03,  4.47887655e-01, -3.04897084e-01],
       [ 1.04440000e+04,  4.89040959e-01, -2.50596617e-01],
       [ 1.10210000e+04,  4.00409138e-01, -3.13103054e-01],
       [ 1.16190000e+04,  4.07420816e-01, -2.85263914e-01],
       [ 2.19000000e+04, -4.12214806e-01,  2.08940828e-01],
       [ 2.25370000e+04, -4.98975523e-01,  1.67496778e-01],
       [ 2.31550000e+04, -4.77820631e-01,  2.20371563e-01],
       [ 2.37520000e+04, -5.23169678e-01,  1.53278570e-01],
       [ 2.43700000e+04, -4.98693129e-01,  1.83100430e-01],
       [ 2.49480000e+04, -4.81070435e-01,  1.26538753e-01],
       [ 2.56250000e+04, -4.37488620e-01,  1.34977284e-01],
       [ 3.45310000e+04,  3.69542352e-01, -3.58167047e-01],
       [ 3.51490000e+04,  4.09735764e-01, -3.62800598e-01],
       [ 3.57070000e+04,  4.68736973e-01, -2.54056253e-01],
       [ 3.63040000e+04,  4.16100912e-01, -2.65995329e-01],
       [ 3.69220000e+04,  3.57261210e-01, -2.26350739e-01],
       [ 4.20820000e+04, -4.77452320e-01, -9.14390576e-02],
       [ 4.27000000e+04, -4.86846726e-01, -1.39294256e-01],
       [ 4.32980000e+04, -4.96724729e-01, -1.06700003e-01],
       [ 4.39150000e+04, -4.70783295e-01, -1.28925318e-01],
       [ 4.45530000e+04, -4.53738649e-01, -6.33537283e-02],
       [ 4.53100000e+04, -3.94338087e-01, -6.70632046e-02],
       [ 4.63060000e+04, -5.08215335e-01, -1.01998595e-01],
       [ 4.69630000e+04, -5.27808811e-01, -1.72019273e-01],
       [ 4.75810000e+04, -5.49700348e-01, -1.00964273e-01],
       [ 4.81790000e+04, -5.13469903e-01, -1.05730309e-01],
       [ 4.88360000e+04, -4.88577968e-01, -4.01604658e-02],
       [ 4.94940000e+04, -4.59431724e-01, -1.68720361e-02],
       [ 5.03500000e+04, -4.67526044e-01, -1.08002137e-02],
       [ 5.11270000e+04, -4.87238890e-01, -1.32566553e-01],
       [ 5.18250000e+04, -4.57476619e-01, -2.08484513e-01],
       [ 5.24620000e+04, -2.73026565e-01, -3.31770086e-01],
       [ 5.65470000e+04, -1.61783015e-01, -4.23114940e-01],
       [ 5.72640000e+04, -1.48434485e-01, -5.33021111e-01],
       [ 5.79020000e+04, -2.12254574e-01, -4.90114175e-01],
       [ 5.84990000e+04, -1.21206171e-01, -5.28022140e-01],
       [ 5.91370000e+04, -1.04632714e-01, -4.94564683e-01],
       [ 5.97540000e+04, -9.85968727e-02, -4.82853826e-01],
       [ 6.04120000e+04, -1.56122710e-01, -4.17992898e-01],
       [ 6.12290000e+04, -1.13644540e-01, -4.34887905e-01],
       [ 6.20460000e+04, -1.91236757e-01, -4.54631921e-01],
       [ 6.27230000e+04, -1.24691462e-01, -4.84814196e-01],
       [ 6.33810000e+04, -1.55356229e-01, -4.96931054e-01],
       [ 6.39980000e+04, -1.27015841e-01, -5.05018178e-01],
       [ 6.46760000e+04, -1.87223964e-01, -4.85621874e-01],
       [ 6.53130000e+04, -1.05207498e-01, -4.50767281e-01],
       [ 6.61100000e+04, -1.60204147e-01, -4.11177747e-01],
       [ 6.69270000e+04, -2.07322639e-01, -4.63809601e-01],
       [ 6.75450000e+04, -2.70777467e-01, -4.93838283e-01],
       [ 6.81420000e+04, -1.29628632e-01, -5.40676865e-01],
       [ 6.87600000e+04, -1.32227688e-01, -5.39917154e-01],
       [ 6.93380000e+04, -4.91868361e-02, -5.35366406e-01],
       [ 6.99750000e+04, -9.72934642e-02, -4.89110154e-01],
       [ 7.06530000e+04, -7.36267644e-02, -4.61310365e-01],
       [ 7.14300000e+04, -1.04173418e-01, -4.96705123e-01],
       [ 7.21670000e+04, -5.21567326e-02, -4.91507244e-01],
       [ 7.28840000e+04, -9.73114685e-02, -4.96716563e-01],
       [ 7.35810000e+04, -4.04279367e-02, -4.90563803e-01],
       [ 7.43390000e+04, -6.51565028e-02, -4.39665309e-01],
       [ 7.60920000e+04, -1.37233625e-01, -4.69486576e-01],
       [ 7.68090000e+04, -1.01556856e-01, -4.85070297e-01],
       [ 7.75260000e+04, -1.19580296e-01, -5.01341787e-01],
       [ 7.82640000e+04, -7.64793978e-02, -4.94769980e-01],
       [ 7.89410000e+04, -1.01239579e-01, -4.37966089e-01],
       [ 7.97980000e+04, -1.37160574e-01, -4.56371590e-01],
       [ 8.04950000e+04, -2.44792252e-01, -4.84929706e-01],
       [ 8.11130000e+04, -1.67253560e-01, -4.86678362e-01],
       [ 8.18100000e+04, -9.65009295e-02, -5.00033199e-01],
       [ 8.25470000e+04,  2.87227721e-02, -4.86220730e-01],
       [ 8.33040000e+04,  1.42307133e-02, -4.75447421e-01],
       [ 8.40610000e+04,  8.32132781e-02, -5.18318321e-01],
       [ 8.46790000e+04,  4.60170631e-02, -5.59048098e-01],
       [ 8.53170000e+04,  1.47076701e-01, -5.34096811e-01],
       [ 8.59540000e+04,  8.85281808e-02, -5.22282730e-01],
       [ 8.66320000e+04,  9.53720951e-02, -4.82875545e-01]])

    waypoint = np.array([[1.18000000e+03, 7.38962860e+01, 8.59697266e+01],
       [3.04800000e+03, 7.28752136e+01, 8.44115372e+01],
       [4.92400000e+03, 7.14424057e+01, 8.31210556e+01],
       [6.81600000e+03, 7.63001862e+01, 8.60178680e+01],
       [8.69300000e+03, 7.25290604e+01, 8.65375671e+01],
       [1.05740000e+04, 6.98909149e+01, 8.52463608e+01],
       [1.24530000e+04, 6.82949524e+01, 8.52412491e+01],
       [1.43570000e+04, 6.82374039e+01, 8.47963028e+01],
       [1.62120000e+04, 6.73553772e+01, 8.52209015e+01],
       [1.80890000e+04, 6.93443451e+01, 8.65693283e+01],
       [1.99740000e+04, 7.24116745e+01, 8.81250076e+01],
       [2.18750000e+04, 6.99019775e+01, 8.41980515e+01],
       [2.37790000e+04, 6.85038071e+01, 8.41305313e+01],
       [2.56600000e+04, 6.64745636e+01, 8.21872406e+01],
       [2.75610000e+04, 6.82560120e+01, 8.22882690e+01],
       [2.94670000e+04, 6.70599670e+01, 8.17957458e+01],
       [3.13730000e+04, 6.48172684e+01, 7.97197952e+01],
       [3.32870000e+04, 6.38634453e+01, 7.86726303e+01],
       [3.51570000e+04, 6.51732330e+01, 7.85293961e+01],
       [3.70590000e+04, 6.46356049e+01, 7.82449722e+01],
       [3.90040000e+04, 6.62113724e+01, 7.96956253e+01],
       [4.08970000e+04, 6.53126831e+01, 7.83882904e+01],
       [4.27720000e+04, 6.52270508e+01, 7.89290695e+01],
       [4.46760000e+04, 6.70317001e+01, 7.96430588e+01],
       [4.65610000e+04, 6.29847527e+01, 7.64031296e+01],
       [4.84450000e+04, 6.58869019e+01, 7.99052200e+01],
       [5.03390000e+04, 6.42322845e+01, 7.85904007e+01],
       [5.22490000e+04, 6.38377266e+01, 8.00705032e+01],
       [5.42070000e+04, 6.86770020e+01, 8.09292831e+01],
       [5.61010000e+04, 6.57306519e+01, 8.02452927e+01],
       [5.80070000e+04, 6.91693039e+01, 7.95818710e+01],
       [5.99130000e+04, 7.18470917e+01, 8.03215790e+01],
       [6.18380000e+04, 7.39593582e+01, 7.96940536e+01],
       [6.37380000e+04, 6.94238434e+01, 7.97254410e+01],
       [6.56400000e+04, 7.12903290e+01, 8.10903244e+01],
       [6.75350000e+04, 7.44353027e+01, 8.32358246e+01],
       [6.94250000e+04, 7.25251999e+01, 8.09420242e+01],
       [7.13630000e+04, 7.27850647e+01, 8.10393066e+01],
       [7.32800000e+04, 6.93235626e+01, 7.89138184e+01],
       [7.52140000e+04, 6.67243347e+01, 7.82338791e+01],
       [7.71330000e+04, 6.51353149e+01, 7.36710281e+01],
       [7.90510000e+04, 6.09761429e+01, 7.62598877e+01],
       [8.09530000e+04, 6.18070564e+01, 7.18401184e+01],
       [8.28600000e+04, 6.28075485e+01, 6.92553635e+01],
       [8.48190000e+04, 6.18284035e+01, 6.92407455e+01],
       [8.67160000e+04, 6.32161255e+01, 6.68957291e+01]])



    #step_positions = correct_positions_mod(rel_positions, waypoint)

    rel_positions_pd = pd.DataFrame(rel_positions, columns=["time", "x", "y"])
    waypoint_pd = pd.DataFrame(waypoint, columns=["time", "x", "y"])

    fig, ax = plt.subplots(2)
    #rel_positions_pd.plot(x="time", y=["x", "y"], marker="o", ax=ax[0], title="rel_pos")
    waypoint_pd.plot(x="time", y=["x", "y"], marker="o", ax=ax[0], title="wifi_xy")

    n_mean = 10
    n_points = int(0.5 * waypoint.shape[0])

    steps_rand_list = []

    for i in range(n_mean):
        rand_ind = sorted(random.sample(range(0, waypoint.shape[0]), n_points))
        waypoint_rand = waypoint[rand_ind, :]
        #print(i)
        step_positions = correct_positions_mod(rel_positions, waypoint_rand)
        if step_positions is not None:
            step_positions_pd = pd.DataFrame(step_positions, columns=["time", "x", "y"])
            step_positions_pd.plot(x="time", y=["x", "y"], marker="o", ax=ax[1], title="steps_xy")

    plt.show()