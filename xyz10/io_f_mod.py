from dataclasses import dataclass

import numpy as np
import pandas as pd
import os
import geopandas as gpd

def calculateDistanceBeacon(txPower, rssi):

    if rssi == 0:
        return -1.0

    ratio = rssi/txPower
    if ratio < 1.0:
        return np.power(ratio, 10)
    else:
        return (0.89976)*np.power(ratio, 7.7095) + 0.111

def calculateDistanceWIFI(rssi0, rssi):

    return np.power(10., rssi0-rssi)

@dataclass
class ReadData:
    header: pd.DataFrame
    acce: pd.DataFrame
    acce_uncali: pd.DataFrame
    gyro: pd.DataFrame
    gyro_uncali: pd.DataFrame
    magn: pd.DataFrame
    magn_uncali: pd.DataFrame
    rotate: pd.DataFrame
    wifi: pd.DataFrame
    ibeacon: pd.DataFrame
    waypoint: pd.DataFrame


def read_data_file(data_filename, isTrain=True):   # todo add custom sensor return values

    corrupt_data = False
    header_keys = ["startTime", "endTime", "SiteID", "FloorId", "FloorName", "FloorIsStandard", "t0"] if isTrain else [
        "startTime", "endTime", "SiteID", "t0"]

    floor_convert = {'1F': 0, '2F': 1, '3F': 2, '4F': 3, '5F': 4,
                     '6F': 5, '7F': 6, '8F': 7, '9F': 8, '10F': 9,
                     'B': -1, 'B1': -1, 'B2': -2, 'B3': -3, 'B4': -4,
                     'F1': 0, 'F2': 1, 'F3': 2, 'F4': 3, 'F5': 4,
                     'F6': 5, 'F7': 6, 'F8': 7, 'F9': 8, 'F10': 9,
                     }
    floor_convert_full = {'1F': 0, '2F': 1, '3F': 2, '4F': 3, '5F': 4,
                          '6F': 5, '7F': 6, '8F': 7, '9F': 8, '10F': 9,
                          'B': -1, 'B1': -1, 'B2': -2, 'B3': -3, 'B4': -4,
                          'BF': -1, 'BM': -1,
                          'F1': 0, 'F2': 1, 'F3': 2, 'F4': 3, 'F5': 4,
                          'F6': 5, 'F7': 6, 'F8': 7, 'F9': 8, 'F10': 9,
                          'L1': 0, 'L2': 1, 'L3': 2, 'L4': 3, 'L5': 4,
                          'L6': 5, 'L7': 6, 'L8': 7, 'L9': 8, 'L10': 9,
                          'L11': 10,
                          'G': 0, 'LG1': 0, 'LG2': 1, 'LM': 0, 'M': 0,
                          'P1': 0, 'P2': 1, }

    time_delta = 0

    acce_len = 5 if isTrain else 6
    acce_uncali_len = 5 if isTrain else 9
    gyro_len = 5 if isTrain else 6
    gyro_uncali_len = 5 if isTrain else 9
    magn_len = 5 if isTrain else 6
    magn_uncali_len = 5 if isTrain else 9
    rotate_len = 5 if isTrain else 6
    wifi_len = 7
    ibeacon_len = 9 if isTrain else 10
    waypoint_len = 4

    header = {}
    acce = []
    acce_uncali = []
    gyro = []
    gyro_uncali = []
    magn = []
    magn_uncali = []
    rotate = []
    wifi = []
    ibeacon = []
    waypoint = []

    def invalid_header(header):

        _invalid = False
        for key in header_keys:
            if key not in header:
                _invalid = True
        if _invalid:
            print("Corrupted header", data_filename)
        return _invalid

    def parse_floorname(file_path):

        words = file_path.split("/")
        _name = words[-2]
        if _name in floor_convert.keys():
            header["FloorName"] = floor_convert[_name]
            header["FloorIsStandard"] = 1
        elif _name in floor_convert_full.keys():
            header["FloorName"] = floor_convert_full[_name]
            header["FloorIsStandard"] = 0

    def parse_header(line_data):

        for i, record in enumerate(line_data):
            if isTrain:
                if "startTime" in record:
                    header["startTime"] = int(record.replace("startTime:", ""))
                elif "endTime" in record:
                    header["endTime"] = int(record.replace("endTime:", ""))
                elif "SiteID" in record:
                    header["SiteID"] = record.replace("SiteID:", "")
                elif "FloorId" in record:
                    header["FloorId"] = record.replace("FloorId:", "")

            else:
                if "startTime" in record:
                    header["startTime"] = int(line_data[i+1])
                elif "endTime" in record:
                    header["endTime"] = int(line_data[i+1])
                elif "SiteID" in record:
                    header["SiteID"] = record.replace("SiteID:", "")



    def parse_line(line_data):  # todo include data for test uncalibrated sensor records?

        time_delta = 0

        if line_data[1] == 'TYPE_ACCELEROMETER':
            acce.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            if len(line_data) > acce_len:
                parse_line([line_data[0]] + line_data[acce_len:])

        elif line_data[1] == 'TYPE_ACCELEROMETER_UNCALIBRATED':
            acce_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            if len(line_data) > acce_uncali_len:
                parse_line([line_data[0]] + line_data[acce_uncali_len:])

        elif line_data[1] == 'TYPE_GYROSCOPE':
            gyro.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            if len(line_data) > gyro_len:
                parse_line([line_data[0]] + line_data[gyro_len:])

        elif line_data[1] == 'TYPE_GYROSCOPE_UNCALIBRATED':
            gyro_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            if len(line_data) > gyro_uncali_len:
                parse_line([line_data[0]] + line_data[gyro_uncali_len:])

        elif line_data[1] == 'TYPE_MAGNETIC_FIELD':
            magn.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            if len(line_data) > magn_len:
                parse_line([line_data[0]] + line_data[magn_len:])

        elif line_data[1] == 'TYPE_MAGNETIC_FIELD_UNCALIBRATED':
            magn_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            if len(line_data) > magn_uncali_len:
                parse_line([line_data[0]] + line_data[magn_uncali_len:])

        elif line_data[1] == 'TYPE_ROTATION_VECTOR':
            rotate.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            if len(line_data) > rotate_len:
                parse_line([line_data[0]] + line_data[rotate_len:])

        elif line_data[1] == 'TYPE_WIFI':
            sys_ts = int(line_data[0])
            ssid = line_data[2]
            bssid = line_data[3]
            rssi = float(line_data[4])
            is5G = int(int(line_data[5]) > 4000)   # 2GHz ~ 2500MHZ and 5Ghz ~ 5700MHz
            delay = int(line_data[0]) - int(line_data[6])
            wifi_data = [sys_ts, ssid, bssid, rssi, is5G, delay]
            wifi.append(wifi_data)
            if len(line_data) > wifi_len:
                parse_line([line_data[0]] + line_data[wifi_len:])

        elif line_data[1] == 'TYPE_BEACON':
            ts = int(line_data[0])
            uuid = line_data[2]
            major = line_data[3]
            minor = line_data[4]
            rssi = float(line_data[6])
            distance = float(line_data[7])
            ibeacon_data = [ts, '_'.join([uuid, major, minor]), rssi, distance]
            ibeacon.append(ibeacon_data)

            if not isTrain:
                time_delta = int(line_data[0]) - int(line_data[9])

            if len(line_data) > ibeacon_len:
                parse_line([line_data[8][-13:]] + line_data[ibeacon_len:])

        elif line_data[1] == 'TYPE_WAYPOINT':
            waypoint.append([int(line_data[0]), float(line_data[2]), float(line_data[3])])
            if len(line_data) > waypoint_len:
                parse_line([line_data[0]] + line_data[waypoint_len:])

        return time_delta

    def shift_time(time_delta):
        if time_delta < 0:  # test and beacon data is available
            _delta = time_delta
            wifi.delay -= time_delta
        elif isTrain:
            _delta = int(header["startTime"])
            t0 = _delta
            acce.time -= t0
            acce_uncali.time -= t0
            gyro.time -= t0
            gyro_uncali.time -= t0
            magn.time -= t0
            magn_uncali.time -= t0
            rotate.time -= t0
            wifi.time -= t0
            ibeacon.time -= t0
            waypoint.time -= t0
            header["startTime"] -= t0
            header["endTime"] -= t0
        else:  # test and beacon data is not available
            _delta = wifi.delay.min()
            wifi.delay -= _delta

        header["t0"] = abs(_delta)

    with open(data_filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line_data in lines:
        try:
            line_data = line_data.strip()
            if not line_data:
                continue
            elif line_data[0] == '#':
                parse_header(line_data[2:].split('\t'))

            line_data = line_data.split('\t')

            _dt = parse_line(line_data)
            time_delta = _dt if time_delta == 0 else time_delta
        except:
            print("Corrupted file", data_filename)
            corrupt_data = True

    parse_floorname(data_filename)

    sensor_columns = ["time", "x_axis", "y_axis", "z_axis"]
    wifi_columns = ["time", "ssid", "bssid", "rssi", "is5G", "delay"]
    beacon_columns = ["time", "id", "rssi", "distance"]
    waypoint_columns = ["time", "x", "y"]

    header = pd.DataFrame([list(header.values())], columns=list(header.keys()))
    acce = pd.DataFrame(acce, columns=sensor_columns)
    acce_uncali = pd.DataFrame(acce_uncali, columns=sensor_columns)
    gyro = pd.DataFrame(gyro, columns=sensor_columns)
    gyro_uncali = pd.DataFrame(gyro_uncali, columns=sensor_columns)
    magn = pd.DataFrame(magn, columns=sensor_columns)
    magn_uncali = pd.DataFrame(magn_uncali, columns=sensor_columns)
    rotate = pd.DataFrame(rotate, columns=sensor_columns)
    wifi = pd.DataFrame(wifi, columns=wifi_columns)
    ibeacon = pd.DataFrame(ibeacon, columns=beacon_columns)
    waypoint = pd.DataFrame(waypoint, columns=waypoint_columns)

    shift_time(time_delta)
    if corrupt_data or invalid_header(header):
        #raise NameError('Corrupted record')
        return None
    else:
        return ReadData(header, acce, acce_uncali, gyro, gyro_uncali, magn, magn_uncali, rotate, wifi, ibeacon, waypoint)

if __name__ == "__main__":

    def calc_records(out):
        return out.acce.shape[0]+out.acce_uncali.shape[0]+out.gyro.shape[0]+out.gyro_uncali.shape[0]\
               +out.magn.shape[0]+out.magn_uncali.shape[0]+out.rotate.shape[0]+out.wifi.shape[0]\
               +out.ibeacon.shape[0]+out.waypoint.shape[0]

    fix_path = "./data_in/test/030b3d94de8acae7c936563d.txt"
    out = read_data_file(fix_path, False)

    if out is not None:
        print(out.header.SiteID.values[0])
