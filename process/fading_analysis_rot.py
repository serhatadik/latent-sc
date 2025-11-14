import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import gpxpy
import gpxpy.gpx
import itertools
from scipy.signal import butter, sosfilt, find_peaks

def find_indices_outside(arr, a, b):
    indices = [i for i, x in enumerate(arr) if not (a < x < b)]
    return indices

folders = ['../data/data/stat_rot/'+name for name in os.listdir('../data/data/stat_rot/') if name.startswith('rot')]

gps_files = [name for name in os.listdir('../data/gps_data/stat_rot/') if name.endswith('Rot.txt')]
gps_datas = []
for gps_file in gps_files:
    gps_datas.append(pd.read_csv(os.path.join('../data/gps_data/stat_rot/', gps_file)))
gps_data = pd.concat(gps_datas, axis=0)

gps_times = np.array(pd.to_datetime(gps_data['date time']) - np.timedelta64(6, 'h'), dtype='datetime64[s]')
latitude = np.array(gps_data['latitude'])
longitude = np.array(gps_data['longitude'])

coords = {gps_times[i]: np.array([longitude[i], latitude[i]]) for i in range(len(gps_times))}


data = {}
for folder in folders:
    for cnt, file in enumerate(os.listdir(folder)):
        if cnt < len(os.listdir(folder))-2:
            time = pd.to_datetime(file.split('-IQ')[0].split('.')[0])
            time = np.datetime64(time).astype('datetime64[s]')
            data[time] = np.load(os.path.join(folder, file))[0]
times = sorted(list(data.keys()))
fig, axs = plt.subplots(1,2)



TX1Ustar_Pow = []
TX2_Pow = []
TX3_Pow = []
TX4_Pow = []
TX5_Pow = []
coordinates = []
coordinates_ustar = []
def animate(i):
    print(times[i], "%i/%i" % (i, len(times)), end='\r')
    axs[0].clear()
    cfmin1 = 3533903750
    cfmax1 = 3533931250

    cfmin2 = 3533945000
    cfmax2 = 3533972500

    cfmin3 = 3533986250
    cfmax3 = 3534013750

    cfmin4 = 3534027500
    cfmax4 = 3534055000

    cfmin5 = 3534068750
    cfmax5 = 3534096250

    fs = 0.22e6
    center = 3.534e9
    nsamps = len(data[times[i]])

    # assume x contains your array of IQ samples
    x = data[times[i]]
    x = x[0:nsamps]
    x = x * np.hamming(len(x))  # apply a Hamming window

    PSD = np.abs(np.fft.fft(x)) ** 2 / (nsamps * fs)
    PSD_log = 10.0 * np.log10(PSD)
    PSD_shifted = np.fft.fftshift(PSD_log)

    f = np.arange(fs/-2.0, fs/2.0, fs/nsamps)  # start, stop, step.  centered around 0 Hz
    f += center  # now add center frequency
    #axs[0].plot(f, PSD_shifted)
    #axs[0].set_ylim(-120, -10)

    idx1 = find_indices_outside(f, cfmin1, cfmax1)
    idx2 = find_indices_outside(f, cfmin2, cfmax2)
    idx3 = find_indices_outside(f, cfmin3, cfmax3)
    idx4 = find_indices_outside(f, cfmin4, cfmax4)
    idx5 = find_indices_outside(f, cfmin5, cfmax5)

    PSD_shifted_1 = PSD_shifted.copy()
    PSD_shifted_1[idx1] = -200

    PSD_shifted_2 = PSD_shifted.copy()
    PSD_shifted_2[idx2] = -200

    PSD_shifted_3 = PSD_shifted.copy()
    PSD_shifted_3[idx3] = -200

    PSD_shifted_4 = PSD_shifted.copy()
    PSD_shifted_4[idx4] = -200

    PSD_shifted_5 = PSD_shifted.copy()
    PSD_shifted_5[idx5] = -200

    arg_m1 = np.argmax(PSD_shifted_1)
    arg_m2 = np.argmax(PSD_shifted_2)
    arg_m3 = np.argmax(PSD_shifted_3)
    arg_m4 = np.argmax(PSD_shifted_4)
    arg_m5 = np.argmax(PSD_shifted_5)

    if times[i] in coords:
        TX1Ustar_Pow.append(PSD_shifted_1[arg_m1])
        coordinates_ustar.append(coords[times[i]])

        TX2_Pow.append(PSD_shifted_2[arg_m2])
        TX3_Pow.append(PSD_shifted_3[arg_m3])
        TX4_Pow.append(PSD_shifted_4[arg_m4])
        TX5_Pow.append(PSD_shifted_5[arg_m5])
        coordinates.append(coords[times[i]])



for i, time in enumerate(times):
    animate(i)
    if i%1000==0:
        print(i/len(times))

np.save("../files_generated_by_process_data_scripts/TX1Ustar_pow_rot.npy", TX1Ustar_Pow)
np.save("../files_generated_by_process_data_scripts/TX2_pow_rot.npy", TX2_Pow)
np.save("../files_generated_by_process_data_scripts/TX3_pow_rot.npy", TX3_Pow)
np.save("../files_generated_by_process_data_scripts/TX4_pow_rot.npy", TX4_Pow)
np.save("../files_generated_by_process_data_scripts/TX5_pow_rot.npy", TX5_Pow)
np.save("../files_generated_by_process_data_scripts/coordinates_rot.npy", coordinates)
np.save("../files_generated_by_process_data_scripts/coordinates_ustar_rot.npy", coordinates_ustar)
