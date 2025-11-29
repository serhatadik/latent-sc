"""NRDZ TIREM Propagation Model."""

import numpy as np
import pandas as pd
import scipy.io as sio
from time import time
try:
    from numba import jit
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
from ctypes import *
from .common import *
from .tirem_params import *
from .plotter import *
import math
from datetime import date

tirem_dll = None


def main():
    """Entry point into tirem code."""
    load_tirem_lib()
    bs_endpoint_name = "USTAR"
    bs_is_tx = 0
    txheight = 1.5
    rxheight = 3
    bs_lon = -111.84167
    bs_lat = 40.76895
    freq = 462.7
    polarz = 'H'
    region = "rectangle"
    #list_of_latlons = [[40.76702, -111.83807], [40.76749, -111.83607], [40.77283, -111.84088], [40.76989, -111.83869], [40.77108, -111.84316], [40.76899, -111.84167]]
    list_of_latlons = [[40.76702, -111.83807], [40.77283, -111.84088]]

    first_elements = [x[0] for x in list_of_latlons]
    second_elements = [x[1] for x in list_of_latlons]

    # Find the minimum and maximum of the first elements
    min_first_element = min(first_elements)
    max_first_element = max(first_elements)

    # Find the minimum and maximum of the second elements
    min_second_element = min(second_elements)
    max_second_element = max(second_elements)

    utm_easting1, utm_northing1, zone_number, zone_letter = wgs84_to_utm(min_second_element, min_first_element, 12)
    utm_easting2, utm_northing2, zone_number, zone_letter = wgs84_to_utm(max_second_element, max_first_element, 12)


    run_date = tirem_pred(bs_endpoint_name, bs_is_tx, txheight, rxheight, bs_lon, bs_lat, freq, polarz, region, list_of_latlons)
    f = np.load(bs_endpoint_name + '_tirem_rssi_' + str(freq) + run_date+".npz")

    plotter([utm_easting1, utm_easting2], [utm_northing1, utm_northing2], f["arr_0"], "Tirem RSSI")
    plotter([utm_easting1, utm_easting2], [utm_northing1, utm_northing2], f["arr_1"], "Distances")

    f = np.load(bs_endpoint_name + '_diff_parameter_' + str(freq) + run_date+".npy")
    plotter([utm_easting1, utm_easting2], [utm_northing1, utm_northing2], f[:, :, 0], "Diffraction Parameter 1")
    plotter([utm_easting1, utm_easting2], [utm_northing1, utm_northing2], f[:, :, 1], "Diffraction Parameter 2")

    f = np.load(bs_endpoint_name + '_diff_param_obstacle_with_max_h_over_r_' + str(freq) + run_date+".npy")
    plotter([utm_easting_min, utm_easting_max], [utm_northing_min, utm_northing_max], f, "Diffraction Parameter for the Obstacle w/ Max h/r")

    f = np.load(bs_endpoint_name + '_elevation_angles_NLOS_' + str(freq) + run_date+".npy")
    plotter([utm_easting_min, utm_easting_max], [utm_northing_min, utm_northing_max], f[:, :, 0], "Elevation Angles NLOS 1")
    plotter([utm_easting_min, utm_easting_max], [utm_northing_min, utm_northing_max], f[:, :, 1], "Elevation Angles NLOS 2")

    f = np.load(bs_endpoint_name + '_elevation_angles_tx_rx_' + str(freq) + run_date+".npy")
    plotter([utm_easting_min, utm_easting_max], [utm_northing_min, utm_northing_max], f, "Elevation Angles LOS (Tx to Rx)")

    f = np.load(bs_endpoint_name + '_LOS_NLOS_' + str(freq) + run_date+".npy")
    plotter([utm_easting_min, utm_easting_max], [utm_northing_min, utm_northing_max], f, "LOS / NLOS")

    f = np.load(bs_endpoint_name + '_number_knife_edges_' + str(freq) + run_date+".npy")
    plotter([utm_easting_min, utm_easting_max], [utm_northing_min, utm_northing_max], f, "Number of Knife Edges")

    f = np.load(bs_endpoint_name + '_number_obstacles_' + str(freq) + run_date+".npy")
    plotter(f, "Number of Blocking Obstructions")

    f = np.load(bs_endpoint_name + '_shadowing_angles_' + str(freq) + run_date + ".npy")
    plotter(f[:, :, 0], "Shadowing Angle 1")
    plotter(f[:, :, 1], "Shadowing Angle 2")


def tirem_pred(bs_endpoint_name, bs_is_tx, txheight, rxheight, bs_lon, bs_lat, freq,
               polarz, region, list_of_latlons,generate_features=1, map_type="fusion", map_filedir="SLCmap_5May2022.mat", gain=0, first_call=0,
               extsn=0,
               refrac=450, conduc=50, permit=81, humid=80, side_len=0.5, sampling_interval=0.5):
    """
    Code for running TIREM and generating features for augmented modeling. Adapted from SLC_TIREM_v3_Python.

    :param bs_endpoint_name: basestation endpoint name e.g. "cbrssdr1-ustar-comp"
    :param bs_is_tx: boolean, is the basestation a transmitter (1) or a receiver (0)
    :param txheight: transmitter height (m) -DO NOT INCLUDE THE BUILDING HEIGHT OR
    ELEVATION, JUST THE TX HEIGHT W.R.TO THE SURFACE IT STAYS ON
    :param rxheight: receiver height (m) -DO NOT INCLUDE THE BUILDING HEIGHT OR
    ELEVATION, JUST THE RX HEIGHT W.R.TO THE SURFACE IT STAYS ON
    :param bs_lon: basestation longitude
    :param bs_lat: basestation latitude
    :param freq: transmit frequency in MHz
    :param polarz: polarization of the wave ('H' for horizontal, 'V' for vertical)
            produced by the Tx antenna
    :param generate_features: boolean, Generate only TIREM predictions (0) / also generate
                       features (1)
    :param map_type: Fusion map ("fusion") or lidar DSM map ("lidar"). The fusion map
              struct has to have the fields "dem" for digital elevation map and
              "hybrid_bldg" for building heights. The lidar map has to have the field
              "data" having combined information of elevations and building heights.
    :param map_filedir: map file directory including the filename and extension
    gain: antenna gain in dB
    :param gain: antenna gain in dB
    :param first_call: boolean, 1 loads the TIREM library for people using it for
               the first time.
    :param extsn: boolean, 0 is false; anything else is true. False = new profile,
           true = extension of last profile terrain
    :param refrac: surface refractivity, range: (200 to 450.0) "N-Units"
    :param conduc: conductivity, range: (0.00001 to 100.0) S/m
    :param permit: relative permittivity of earth surface, range: (1 to 1000)
    :param humid: humidity, units g/m^3, range: (0 to 110.0)
    :param side_len: grid_cell side length (m)
    :param sampling_interval: sampling interval along the tx-rx link. e.g. 0.5
                       means a given grid cell is sampled twice. Side length
                       of 0.5 and sampling interval of 0.5 mean that the
                       Tx-Rx horizontal array step size 0.5*0.5 = 0.25 m.
    :return: run_date, the date the script completed its run.
    Saves tirem rssi and generated features to the current folder.
    Serhat Tadik, Aashish Gottipati, Michael A. Varner, Gregory D. Durgin
    """


    # load the map
    map_struct = sio.loadmat(map_filedir)['SLC']

    # Define a new struct named SLC
    SLC = map_struct[0][0]
    column_map = dict(zip([name for name in SLC.dtype.names], [i for i in range(len(SLC.dtype.names))]))

    bs_x, bs_y, idx = lon_lat_to_grid_xy(np.array([bs_lon]), np.array([bs_lat]), SLC,
                                         column_map)  # Longitude - latitude
    nrows = int(SLC[column_map['nrows']])
    ncols = int(SLC[column_map['ncols']])
    pred_coords_x = []
    pred_coords_y = []
    if region == "set":
        for i in list_of_latlons:
            rx, ry, idx = lon_lat_to_grid_xy(np.array([i[1]]), np.array([i[0]]), SLC,
                                                 column_map)  # Longitude - latitude
            pred_coords_x.append(rx)
            pred_coords_y.append(ry)
        print(pred_coords_x)
        print(pred_coords_y)
    elif region == "rectangle":
        assert len(list_of_latlons)==2
        rx1, ry1, idx = lon_lat_to_grid_xy(np.array([list_of_latlons[0][1]]), np.array([list_of_latlons[0][0]]), SLC,
                                             column_map)  # Longitude - latitude
        rx2, ry2, idx = lon_lat_to_grid_xy(np.array([list_of_latlons[1][1]]), np.array([list_of_latlons[1][0]]), SLC,
                                             column_map)  # Longitude - latitude
        pred_coords_x = range(min([rx1, rx2]), max([rx1, rx2]))
        pred_coords_y = range(min([ry1, ry2]), max([ry1, ry2]))
    elif region =="all":
        pred_coords_x = range(ncols)
        pred_coords_y = range(nrows)
    # parameters
    input_params = {'bs_endpoint_name': bs_endpoint_name, 'bs_is_tx': bs_is_tx, 'txheight': txheight,
                    'rxheight': rxheight,
                    'bs_lon': bs_lon, 'bs_lat': bs_lat, 'bs_x': bs_x, 'bs_y': bs_y, 'freq': freq,
                    'polarz': polarz + '   ', 'generate_features': generate_features,
                    'map_type': map_type, 'map_filedir': map_filedir, 'gain': gain, 'first_call': first_call,
                    'extsn': extsn,
                    'refrac': refrac, 'conduc': conduc, 'permit': permit, 'humid': humid, 'side_len': side_len,
                    'sampling_interval': sampling_interval}

    params = Params(**input_params)

    ## Get tirem loss for all points

    # determine the tx/rx grid (raster) coordinates
    if bs_is_tx:
        tx = np.array([params.bs_x, params.bs_y])
    else:
        rx = np.array([params.bs_x, params.bs_y])

    # generate the map
    if map_type == "fusion":
        slc_map = SLC[column_map['dem']] + 0.3048 * SLC[column_map['hybrid_bldg']]
    elif map_type == "lidar":
        slc_map = SLC[column_map['data']]

    # the gain
    EIRP = params.gain

    ## Run TIREM
    t = time()

    # Initialize variables
    wavelength = 300 / params.freq
    tirem_rssi = np.ones((nrows, ncols)) * np.NaN
    distances = np.ones((nrows, ncols)) * np.NaN

    if generate_features:
        # line-of-sight / non-line-of-sight
        LOS = np.ones((nrows, ncols))

        # number of blocking obstructions
        number_obstacles = np.zeros((nrows, ncols))

        # number of knife edges
        number_knife_edges = np.zeros((nrows, ncols))

        # elevation angle between Tx & Rx with no consideration of obstacles in between
        elevation_angles_tx_rx = np.zeros((nrows, ncols))

        # elevation angles considering the obstacles in between Tx & Rx
        elevation_angles_NLOS = np.zeros((nrows, ncols, 2))

        # shadowing angles for the first and last knife-edges
        first_last_shadowing_angles = np.zeros((nrows, ncols, 2))

        # diffraction angles for the first and last knife-edges
        first_last_diffraction_angles = np.zeros((nrows, ncols, 2))

        # Fresnel-Kirchhoff diffraction parameter
        diff_parameter = np.zeros((nrows, ncols, 2))

        # Fresnel-Kirchhoff diffraction parameter for the obstacle with the
        # largest h (height of the obstacle above the tx-rx link) /r (fresnel
        # zone at that point) ratio
        diff_param_obstacle_with_max_h_over_r = np.zeros((nrows, ncols))

    # Determine tx / rx height from the map
    if bs_is_tx:
        tx_elevation = slc_map[tx[1], tx[0]]
    else:
        rx_elevation = slc_map[rx[1], rx[0]]
    # Loop through the map pixels
        for _, a in enumerate(pred_coords_x):

            # Display progress
            if np.mod(a, 50) == 0:
                print("Progress: ")
                print(str(100 * _ / len(pred_coords_x)) + '   %')

            for b in pred_coords_y:

                flag = 0
                cntr = 0
                if bs_is_tx:
                    rx = np.array([a, b]) + np.array([1, 1])
                    rx_elevation = slc_map[b, a]
                else:
                    tx = np.array([a, b]) + np.array([1, 1])
                    tx_elevation = slc_map[b, a]
                if (rx == tx).all() == 0:
                    # build arrays and calculate TIREM's predictions
                    [d_array, e_array] = build_arrays(side_len, sampling_interval, tx, rx, slc_map)
                    tirem_rssi[b, a] = EIRP - get_tirem_loss(d_array, e_array, params)
                    dir_ = rx - tx
                    distances[b, a] = np.linalg.norm(dir_, 2) * side_len

                    if generate_features:
                        # Get the nonzero elements of the distance and elevation arrays

                        nonzero_e_array_indices = [e_array != 0]
                        d_array_nonzero = d_array[tuple(nonzero_e_array_indices)]
                        e_array_nonzero = e_array[tuple(nonzero_e_array_indices)]

                        # Calculate the total Rx and Tx heights and the slope between them
                        rx_total_height = rx_elevation + params.rxheight
                        tx_total_height = tx_elevation + params.txheight
                        slope = (rx_total_height - tx_total_height) / max(d_array_nonzero)

                        # Elevation angle
                        elevation_angles_tx_rx[b, a] = 90 - math.degrees(math.atan(slope))

                        # Blocking obstruction information, bo_info, initialization
                        bo_info = np.array([[0, 0, 0]])

                        for i in range(len(e_array_nonzero) - 1):
                            cntr = cntr + 1
                            # if there is a blockage, LOS = 0, (NLOS = 1)
                            if (d_array_nonzero[i] * slope + tx_total_height) < (e_array_nonzero[i]):
                                LOS[b, a] = 0
                                # if it's the first obstacle, calculate the diffraction parameter
                                if flag == 0:
                                    h = e_array_nonzero[i] - (d_array_nonzero[i] * slope + tx_total_height)
                                    d1 = max(np.sqrt((d_array_nonzero[i] * slope) ** 2 + d_array_nonzero[i] ** 2), 0.25)
                                    d2 = max(np.sqrt(((d_array_nonzero[-1] - d_array_nonzero[i]) * slope) ** 2 + (d_array_nonzero[-1] - d_array_nonzero[i]) ** 2), 0.25)
                                    diff_parameter[b, a, 0] = h * np.sqrt(2 * (d1 + d2) / (wavelength * d1 * d2))
                                # calculate the number of obstacles and fill the bo_info with distance, elevation, h, and
                                # r(Fresnel zone) information of obstacles
                                if flag == 0 or cntr > 200:
                                    number_obstacles[b, a] = number_obstacles[b, a] + 1
                                    cntr = 0
                                    h = e_array_nonzero[i] - (d_array_nonzero[i] * slope + tx_total_height)
                                    r = np.sqrt(wavelength * d_array_nonzero[i] * (d_array_nonzero[-1] - d_array_nonzero[i])) * np.sqrt(1 + slope ** 2) / d_array_nonzero[-1]
                                    bo_info = np.append(bo_info, [[d_array_nonzero[i], h, r]], 0)
                                    if flag == 0:
                                        bo_info = np.delete(bo_info, 0, 0)

                                    flag = 1


                        # find the obstacle with largest h / r ratio and calculate the diffraction parameter for it
                        if (bo_info == np.array([[0, 0, 0]])).all() == 0:
                            idxx = (bo_info[:, 1] / bo_info[:, 2]) == max(bo_info[:, 1] / bo_info[:, 2])
                            d1_ = max(bo_info[idxx, 0], 0.25)
                            h_ = bo_info[idxx, 1]
                            diff_param_obstacle_with_max_h_over_r[b, a] = h_ * np.sqrt(2 * (d_array_nonzero[-1] * (np.sqrt(1 + slope ** 2))) / (wavelength * d1_ * (d_array_nonzero[-1] - d1_) * (1 + slope ** 2)))

                        # knife edge information, ke_info, initialization
                        ke_info = np.array([[0, 0]])
                        flag = 0
                        cntr = 0
                        d_ke = 0
                        for i in range(len(e_array_nonzero) - 1):
                            cntr = cntr + 1
                            # different from blocking obstacle, change the slope each time you come across an obstacle
                            if ((d_array_nonzero[i] - d_ke) * slope + tx_total_height) < (e_array_nonzero[i]):
                                # calculate number of knife edges
                                if flag == 0 or cntr > 200:
                                    number_knife_edges[b, a] += 1
                                    # if it's the first ke, calculate the shadowing angle
                                    if flag == 0:
                                        first_last_shadowing_angles[b, a, 0] = math.degrees(math.atan(
                                            (e_array_nonzero[i] - tx_total_height) / d_array_nonzero[i])) - math.degrees(
                                            math.atan(slope))

                                    # update ke_info with distance and elevation information
                                    ke_info = np.append(ke_info, [[d_array_nonzero[i], e_array_nonzero[i]]], 0)
                                    if flag == 0:
                                        ke_info = np.delete(ke_info, 0, 0)
                                    # update parameters
                                    d_ke = d_array_nonzero[i]
                                    tx_total_height = e_array_nonzero[i]
                                    slope = (rx_total_height - tx_total_height) / (max(d_array_nonzero) - d_array_nonzero[i])

                                    cntr = 0
                                    flag = 1

                        # redefine the original slope in case it changes in the previous loop
                        rx_total_height = rx_elevation + params.rxheight
                        tx_total_height = tx_elevation + params.txheight
                        slope = (rx_total_height - tx_total_height) / max(d_array_nonzero)

                        # if there is at least one ke, calculate NLOS elevation angles, shadowing angle for the
                        # last ke, and diffraction angles.
                        if (ke_info == np.array([[0, 0]])).all() == 0:

                            h = ke_info[-1, 1] - ((ke_info[-1, 0] - d_array_nonzero[-1]) * slope + rx_total_height)
                            d1 = max(np.sqrt(((ke_info[-1, 0] - d_array_nonzero[-1]) * slope) ** 2 + (ke_info[-1, 0] - d_array_nonzero[-1]) ** 2), 0.25)
                            d2 = max(np.sqrt(((d_array_nonzero[0] - ke_info[-1, 0]) * slope) ** 2 + (d_array_nonzero[0] - ke_info[-1, 0]) ** 2), 0.25)

                            diff_parameter[b, a, 1] = h * np.sqrt(2 * (d1 + d2) / (wavelength * d1 * d2))
                            first_last_shadowing_angles[b, a, 1] = math.degrees(math.atan((ke_info[-1, 1] - rx_total_height) / (d_array_nonzero[-1] - ke_info[-1, 0]))) + math.degrees(math.atan(slope))
                            elevation_angles_NLOS[b, a, 0] = 90 - math.degrees(math.atan((ke_info[0, 1] - tx_total_height) / max((ke_info[0, 0] - d_array_nonzero[0]),0.25)))
                            elevation_angles_NLOS[b, a, 1] = 90 - math.degrees(math.atan((ke_info[-1, 1] - rx_total_height) / max((d_array_nonzero[-1] - ke_info[-1, 0]),0.25)))

                            if ke_info.shape[0] == 1:
                                slope_ke_to_rx = (ke_info[0, 1] - rx_total_height) / (d_array_nonzero[-1] - ke_info[0, 0])
                                first_last_diffraction_angles[b, a, 0] = math.degrees(math.atan((ke_info[0, 1] - tx_total_height) / ke_info[0, 0])) + math.degrees(math.atan(slope_ke_to_rx))
                                first_last_diffraction_angles[b, a, 1] = first_last_diffraction_angles[b, a, 0]

                            if ke_info.shape[0] != 1:

                                slope_1 = (ke_info[0, 1] - tx_total_height) / ke_info[0, 0]
                                slope_2 = (ke_info[0, 1] - ke_info[1, 1]) / (ke_info[1, 0] - ke_info[0, 0])
                                first_last_diffraction_angles[b, a, 0] = math.degrees(math.atan(slope_1)) + math.degrees(math.atan(slope_2))

                                # if the encountered obstacle isn't actually an obstacle that the waves would diffract from
                                # (e.g.the first encountered obstacle doesn't block the link between the Tx and the second
                                # obstacle), correct the variables
                                cnt_ke = 0
                                cnt2 = 0
                                while cnt_ke < ke_info.shape[0]:
                                    cnt2 = cnt2 + 1
                                    cnt_ke = cnt_ke + 1
                                    if cnt_ke == 1:
                                        slope_1 = (ke_info[cnt_ke-1, 1] - tx_total_height) / ke_info[cnt_ke-1, 0]
                                    else:
                                        slope_1 = (ke_info[cnt_ke-1, 1] - ke_info[cnt_ke - 2, 1]) / (ke_info[cnt_ke-1, 0] - ke_info[cnt_ke - 2, 0])

                                    if cnt_ke == ke_info.shape[0]:
                                        slope_2 = (ke_info[cnt_ke-1, 1] - rx_total_height) / (d_array_nonzero[-1] - ke_info[cnt_ke-1, 0])
                                    else:
                                        slope_2 = (ke_info[cnt_ke-1, 1] - ke_info[cnt_ke, 1]) / (ke_info[cnt_ke , 0] - ke_info[cnt_ke-1, 0])
                                    cnt = cnt_ke
                                    idx = 0
                                    while math.degrees(math.atan(slope_1)) < -1 * math.degrees(math.atan(slope_2)):
                                        idx = idx + 1
                                        if ke_info.shape[0] == cnt + 1:

                                            if ke_info.shape[0]- 1 - idx == 0:
                                                slope_1 = (ke_info[-1, 1] - tx_total_height) / ke_info[-1, 0]
                                            else:
                                                slope_1 = (ke_info[-1, 1] - ke_info[-2 - idx, 1]) / (ke_info[-1, 0] - ke_info[- 2 - idx, 0])

                                            slope_2 = (ke_info[-1, 1] - rx_total_height) / (d_array_nonzero[-1] - ke_info[-1, 0])
                                        else:
                                            if cnt2 == 1:
                                                slope_1 = (ke_info[cnt, 1] - tx_total_height) / ke_info[cnt, 0]
                                            else:
                                                slope_1 = (ke_info[cnt, 1] - ke_info[cnt_ke - idx - 1, 1]) / (ke_info[cnt, 0] - ke_info[cnt_ke - idx - 1, 0])

                                            slope_2 = (ke_info[cnt, 1] - ke_info[cnt + 1, 1]) / (
                                                        ke_info[cnt + 1, 0] - ke_info[cnt , 0])

                                        if cnt2 == 1:
                                            h__ = ke_info[cnt, 1] - (ke_info[cnt, 0] * slope + tx_total_height)
                                            d1__ = math.sqrt((ke_info[cnt, 0] * slope) ** 2 + ke_info[cnt, 0] ** 2)
                                            d2__ = math.sqrt(((d_array_nonzero[-1] - ke_info[cnt, 0]) * slope) ** 2 + (d_array_nonzero[-1] - ke_info[cnt , 0]) ** 2)
                                            diff_parameter[b, a, 0] = h__ * math.sqrt(2 * (d1__ + d2__) / (wavelength * d1__ * d2__))

                                            first_last_diffraction_angles[b, a, 0] = math.degrees(math.atan(slope_1)) + math.degrees(math.atan(slope_2))
                                            first_last_shadowing_angles[b, a, 0] = math.degrees(math.atan(slope_1))

                                        cnt = cnt + 1
                                        cnt_ke = cnt_ke + 1
                                        number_knife_edges[b, a] = number_knife_edges[b, a] - 1
                                        if number_knife_edges[b, a] < 0:
                                            print(number_knife_edges[b, a])

                                if first_last_diffraction_angles[b, a, 0] < 0:
                                    print(first_last_diffraction_angles[b, a, 0])


                                slope_last_ke_to_rx = (ke_info[-1, 1] - rx_total_height) / (d_array_nonzero[-1] - ke_info[-1, 0])
                                slope_ke_beforelast_to_ke_last = (ke_info[-2, 1] - ke_info[-1, 1]) / (ke_info[-1, 0] - ke_info[-2, 0])
                                first_last_diffraction_angles[b, a, 1] = math.degrees(math.atan(slope_last_ke_to_rx)) - math.degrees(math.atan(slope_ke_beforelast_to_ke_last))
    if region != "all":
        tirem_rssi = tirem_rssi[min(pred_coords_y):max(pred_coords_y), min(pred_coords_x):max(pred_coords_x)]
        LOS = LOS[min(pred_coords_y):max(pred_coords_y), min(pred_coords_x):max(pred_coords_x)]
        number_obstacles = number_obstacles[min(pred_coords_y):max(pred_coords_y), min(pred_coords_x):max(pred_coords_x)]
        number_knife_edges = number_knife_edges[min(pred_coords_y):max(pred_coords_y), min(pred_coords_x):max(pred_coords_x)]
        elevation_angles_tx_rx = elevation_angles_tx_rx[min(pred_coords_y):max(pred_coords_y), min(pred_coords_x):max(pred_coords_x)]
        elevation_angles_NLOS = elevation_angles_NLOS[min(pred_coords_y):max(pred_coords_y), min(pred_coords_x):max(pred_coords_x)]
        first_last_shadowing_angles = first_last_shadowing_angles[min(pred_coords_y):max(pred_coords_y), min(pred_coords_x):max(pred_coords_x)]
        first_last_diffraction_angles = first_last_diffraction_angles[min(pred_coords_y):max(pred_coords_y), min(pred_coords_x):max(pred_coords_x)]
        diff_parameter = diff_parameter[min(pred_coords_y):max(pred_coords_y), min(pred_coords_x):max(pred_coords_x)]
        diff_param_obstacle_with_max_h_over_r = diff_param_obstacle_with_max_h_over_r[min(pred_coords_y):max(pred_coords_y), min(pred_coords_x):max(pred_coords_x)]
    elapsed = time() - t
    print("Elapsed time is:" + str(elapsed))
    today = date.today()

    np.savez(bs_endpoint_name + '_tirem_rssi_' + str(freq) + today.strftime("%b-%d-%Y") + '.npz', tirem_rssi, distances,
             params)

    if generate_features:
        np.save(bs_endpoint_name + '_LOS_NLOS_' + str(freq) + today.strftime("%b-%d-%Y") + '.npy', LOS)
        np.save(bs_endpoint_name + '_number_obstacles_' + str(freq) + today.strftime("%b-%d-%Y") + '.npy',
                number_obstacles)
        np.save(bs_endpoint_name + '_number_knife_edges_' + str(freq) + today.strftime("%b-%d-%Y") + '.npy',
                number_knife_edges)
        np.save(bs_endpoint_name + '_elevation_angles_tx_rx_' + str(freq) + today.strftime("%b-%d-%Y") + '.npy',
                elevation_angles_tx_rx)
        np.save(bs_endpoint_name + '_elevation_angles_NLOS_' + str(freq) + today.strftime("%b-%d-%Y") + '.npy',
                elevation_angles_NLOS)
        np.save(bs_endpoint_name + '_shadowing_angles_' + str(freq) + today.strftime("%b-%d-%Y") + '.npy',
                first_last_shadowing_angles)
        np.save(bs_endpoint_name + '_diffraction_angles_' + str(freq) + today.strftime("%b-%d-%Y") + '.npy',
                first_last_diffraction_angles)
        np.save(bs_endpoint_name + '_diff_parameter_' + str(freq) + today.strftime("%b-%d-%Y") + '.npy', diff_parameter)
        np.save(bs_endpoint_name + '_diff_param_obstacle_with_max_h_over_r_' + str(freq) + today.strftime(
            "%b-%d-%Y") + '.npy', diff_param_obstacle_with_max_h_over_r)

    return today.strftime("%b-%d-%Y")


def load_tirem_lib():
    """Loads tirem DLL"""
    global tirem_dll

    import os
    dll_path = os.path.join(os.path.dirname(__file__), "TIREM320DLL.dll")
    try:
        tirem_dll = CDLL(dll_path)
    except OSError:
        try:
            tirem_dll = WinDLL(dll_path)
        except OSError:
            print(f'ERROR! Failed to load TIREM DLL from {dll_path}')


def call_tirem_loss(d_array, e_array, params):
    """Sets up data for Tirem DLL call."""
    # Load DLL
    if tirem_dll is None:
        load_tirem_lib()

    # initialize the pointer and data for each argument
    # inputs just set to some number in their valid range
    TANTHT = pointer(c_float(params.txheight))  # 0 - 30, 000m
    RANTHT = pointer(c_float(params.rxheight))
    PROPFQ = pointer(c_float(params.freq))  # 1 to 20, 000 MHz

    # next three values characterize the shape of terrain
    NPRFL = pointer(c_int32(d_array.shape[0]))  # number of points in array MAYBE TGUS

    HPRFL = e_array.astype(np.float32).ctypes.data_as(POINTER(c_float))  # array of above (mean) sea level heights
    XPRFL = d_array.astype(np.float32).ctypes.data_as(POINTER(c_float))  # array of great circles distances between
    # points and start

    EXTNSN = pointer(c_int32(params.extsn))  # boolean, 0 is false
    # anything else is true. False = new profile, true = extension of last profile terrain
    # Haven't been able to figure out what the extension flag actually does

    REFRAC = pointer(c_float(params.refrac))  # Surface refractivity  200 to 450.0 "N-Units"
    CONDUC = pointer(c_float(params.conduc))  # 0.00001 to 100.0 S/m
    PERMIT = pointer(c_float(params.permit))  # Relative permittivity of earth surface  1 to 1000
    HUMID = pointer(c_float(params.humid))  # Units g/m^3   0 to 110.0
    polar_ascii = np.array([ord(c) for c in params.polarz])
    POLARZ = polar_ascii.astype(np.uint8).ctypes.data_as(POINTER(c_void_p))

    # output starts here, I just intialize them all to 0
    VRSION = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8).ctypes.data_as(POINTER(c_void_p))
    MODE = np.array([0, 0, 0, 0], dtype=np.uint8).ctypes.data_as(POINTER(c_void_p))
    LOSS = pointer(c_float(0))
    FSPLSS = pointer(c_float(0))

    tirem_dll.CalcTiremLoss(TANTHT, RANTHT, PROPFQ, NPRFL, HPRFL, XPRFL, EXTNSN, REFRAC, CONDUC,
                            PERMIT, HUMID, POLARZ, VRSION, MODE, LOSS, FSPLSS)
    return LOSS.contents.value


def get_tirem_loss(d_array, e_array, params):
    """Returns TIREM loss.

    Usage: loss = get_tirem_loss(d_array, e_array, params)

    inputs: d_array - distance array
            e_array - elevation array
            params - transmitter parameters

    output:   loss - estimated propagation loss
    """
    return call_tirem_loss(d_array, e_array, params)


if __name__ == '__main__':
    main()
