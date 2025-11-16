import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import gpxpy
import gpxpy.gpx
import itertools
from scipy.signal import butter, sosfilt, find_peaks
from geopy import distance

def find_indices_outside(arr, a, b):
    indices = [i for i, x in enumerate(arr) if not (a < x < b)]
    return indices

gps_serhat_files = ['../raw_data/gps_data_serhat/'+name for name in os.listdir('../raw_data/gps_data_serhat/') if name.startswith('2023') ]

coords_serhat = dict()

for gps_serhat_file in gps_serhat_files:
    gpx_file = open(gps_serhat_file, 'r')
    gpx = gpxpy.parse(gpx_file)
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                #print(type(point.time))
                #print('Point at ({0},{1}) -> {2}'.format(point.latitude, point.longitude, np.array(pd.to_datetime(point.time) - np.timedelta64(6, 'h'), dtype='datetime64[s]')))
                coords_serhat[np.datetime64(pd.to_datetime(point.time) - np.timedelta64(6, 'h')).astype('datetime64[s]')] = np.array([point.longitude, point.latitude])

folders = ['../raw_data/walking/'+name for name in os.listdir('../raw_data/walking/') if name.startswith('samples_20') ]
folders += ['../raw_data/driving/'+name for name in os.listdir('../raw_data/driving/') if name.startswith('samples_20') ]
gps_files = [name for name in os.listdir('../raw_data/gps_data/all_gps_data/') if name.endswith('.txt')]
gps_datas = []
for gps_file in gps_files:
    gps_datas.append(pd.read_csv(os.path.join('../raw_data/gps_data/all_gps_data/', gps_file)))
gps_data = pd.concat(gps_datas, axis=0)

gps_times = pd.to_datetime(gps_data['date time'], format='mixed') - pd.Timedelta(hours=6)
gps_times = np.array(gps_times, dtype='datetime64[s]')

latitude = np.array(gps_data['latitude'])
longitude = np.array(gps_data['longitude'])

coords = {gps_times[i]: np.array([longitude[i], latitude[i]]) for i in range(len(gps_times))}



for k in coords_serhat.keys():
    if k not in coords.keys():
        if coords_serhat[k][1]>40:
            coords[k] = coords_serhat[k]
            longitude = np.append(longitude, coords_serhat[k][0])
            latitude = np.append(latitude, coords_serhat[k][1])

def get_metadata_from_folder(folder_name):
    if 'walking' in folder_name:
        return 'walking'
    elif 'driving' in folder_name:
        return 'driving'
    else:
        return 'unknown'  # or any default value you prefer

data = {}
meta = []
for folder in folders:
    print(len(os.listdir(folder)))
    for cnt, file in enumerate(os.listdir(folder)):
        if cnt < len(os.listdir(folder))-2:
            time = pd.to_datetime(file.split('-IQ')[0].split('.')[0])
            time = np.datetime64(time).astype('datetime64[s]')
            data[time] = np.load(os.path.join(folder, file))[0]
            meta.append(get_metadata_from_folder(folder))
times = sorted(list(data.keys()))
fig, axs = plt.subplots(1,2)

tx1_ebc = np.load("files_generated_by_process_data_scripts/TX1EBC_pow_test.npy")
tx2_ustar = np.load("files_generated_by_process_data_scripts/TX1Ustar_pow_test.npy")
tx3 = np.load("files_generated_by_process_data_scripts/TX2_pow_test.npy")
tx4 = np.load("files_generated_by_process_data_scripts/TX3_pow_test.npy")
tx5 = np.load("files_generated_by_process_data_scripts/TX4_pow_test.npy")
tx6 = np.load("files_generated_by_process_data_scripts/TX5_pow_test.npy")

coord = np.load("files_generated_by_process_data_scripts/coordinates_test.npy")
coord_ebc = np.load("files_generated_by_process_data_scripts/coordinates_ebc_test.npy")
coord_ustar = np.load("files_generated_by_process_data_scripts/coordinates_ustar_test.npy")

ebcdd = (40.76702, -111.83807)
guesthousedd = (40.76749, -111.83607)
mariodd = (40.77283, -111.84088)
morandd = (40.76989, -111.83869)
wasatchdd = (40.77108, -111.84316)
ustar = (40.76899, -111.84167)

dist1 = []
dist2 = []
dist3 = []
dist4 = []
dist5 = []
dist6 = []

for c in coord:
    dist3.append(distance.distance(tuple(c[::-1]), guesthousedd).m)
    dist4.append(distance.distance(tuple(c[::-1]), mariodd).m)
    dist5.append(distance.distance(tuple(c[::-1]), morandd).m)
    dist6.append(distance.distance(tuple(c[::-1]), wasatchdd).m)

for c in coord_ebc:
    dist1.append(distance.distance(tuple(c[::-1]), ebcdd).m)

for c in coord_ustar:
    dist2.append(distance.distance(tuple(c[::-1]), ustar).m)


ind_rem = []
for i in range(len(tx2_ustar)):
    if dist2[i] < 10**2.7 and tx2_ustar[i] < -88:
        ind_rem.append(i)

ind_rem = np.array(ind_rem)

mask = ~np.isin(np.arange(tx2_ustar.size), ind_rem)

# Apply the mask to filter the arrays
tx2_ustar = np.array(tx2_ustar)[mask]
dist2 = np.array(dist2)[mask]
import json

# Initialize the JSON structure
structured_data = {}

transmitters = {
    "ebcdd": (40.76702, -111.83807),
    "guesthousedd": (40.76749, -111.83607),
    "mariodd": (40.77283, -111.84088),
    "morandd": (40.76989, -111.83869),
    "wasatchdd": (40.77108, -111.84316),
    "ustar": (40.76899, -111.84167)
}

cnt = 0
for i, time in enumerate(times):
    time_str = str(time)
    if i >= len(tx1_ebc) and i >= len(tx2_ustar) and i >= len(tx3) and i >= len(tx4) and i >= len(tx5) and i >= len(tx6):
        continue
    structured_data[time_str] = {
        "pow_rx_tx": [],
        "metadata": [meta[i]]
    }

    # Add power measurement and corresponding coordinates to rx_data

    if i < len(tx2_ustar) + len(tx1_ebc):
        if times[i].astype('datetime64[D]').item().day == 27 and i < len(tx1_ebc):
            structured_data[time_str]["pow_rx_tx"].append([tx1_ebc[i], coord_ebc[i][1], coord_ebc[i][0], transmitters["ebcdd"][0], transmitters["ebcdd"][1]])
        elif times[i].astype('datetime64[D]').item().day != 27:
            if cnt==0:
                cnt += np.where(coord_ustar[:, 0]==coord[i][0])[0][0]
            if mask[cnt] == 1:
                structured_data[time_str]["pow_rx_tx"].append(
                    [tx2_ustar[cnt], coord_ustar[cnt][1], coord_ustar[cnt][0],
                     transmitters["ustar"][0], transmitters["ustar"][1]])
            cnt += 1
    if i < len(tx3):
        structured_data[time_str]["pow_rx_tx"].append([tx3[i], coord[i][1], coord[i][0], transmitters["guesthousedd"][0], transmitters["guesthousedd"][1]])
    if i < len(tx4):
        structured_data[time_str]["pow_rx_tx"].append([tx4[i], coord[i][1], coord[i][0], transmitters["mariodd"][0], transmitters["mariodd"][1]])
    if i < len(tx5):
        structured_data[time_str]["pow_rx_tx"].append([tx5[i], coord[i][1], coord[i][0], transmitters["morandd"][0], transmitters["morandd"][1]])
    if i < len(tx6):
        structured_data[time_str]["pow_rx_tx"].append([tx6[i], coord[i][1], coord[i][0], transmitters["wasatchdd"][0], transmitters["wasatchdd"][1]])

# Export to JSON
with open('data.json', 'w') as json_file:
    json.dump(structured_data, json_file, indent=4)