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

gps_serhat_files = ['../data/gps_data_serhat/'+name for name in os.listdir('../data/gps_data_serhat/') if name.startswith('2023') ]

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

folders = ['../data/data/walking/'+name for name in os.listdir('../data/data/walking/') if name.startswith('samples_20') ]
folders += ['../data/data/driving/'+name for name in os.listdir('../data/data/driving/') if name.startswith('samples_20') ]
gps_files = [name for name in os.listdir('../data/gps_data/all_gps_data/') if name.endswith('.txt')]
gps_datas = []
for gps_file in gps_files:
    gps_datas.append(pd.read_csv(os.path.join('../data/gps_data/all_gps_data/', gps_file)))
gps_data = pd.concat(gps_datas, axis=0)

gps_times = np.array(pd.to_datetime(gps_data['date time']) - np.timedelta64(6, 'h'), dtype='datetime64[s]')
latitude = np.array(gps_data['latitude'])
longitude = np.array(gps_data['longitude'])

coords = {gps_times[i]: np.array([longitude[i], latitude[i]]) for i in range(len(gps_times))}
#print(len(coords.keys()))


for k in coords_serhat.keys():
    if k not in coords.keys():
        if coords_serhat[k][1]>40:
            coords[k] = coords_serhat[k]
            longitude = np.append(longitude, coords_serhat[k][0])
            latitude = np.append(latitude, coords_serhat[k][1])


data = {}
for folder in folders:
    print(len(os.listdir(folder)))
    for cnt, file in enumerate(os.listdir(folder)):
        if cnt < len(os.listdir(folder))-2:
            time = pd.to_datetime(file.split('-IQ')[0].split('.')[0])
            time = np.datetime64(time).astype('datetime64[s]')
            data[time] = np.load(os.path.join(folder, file))[0]
times = sorted(list(data.keys()))
fig, axs = plt.subplots(1,2)

print(len(data.keys()))
'''
def animate(i):
    print(times[i], "%i/%i" % (i, len(times)), end='\r')
    axs[0].clear()
    axs[0].plot(3.534e9+ np.fft.fftshift(np.fft.fftfreq(len(data[times[i]]), 1/0.22e6)), 10.0 * np.log10(abs( np.fft.fftshift(np.fft.fft( data[times[i]])))))
    axs[0].set_ylim(-20,40)

    axs[1].clear()
    axs[1].scatter(longitude, latitude, marker='.', )
    if times[i] in coords:
        axs[1].scatter(*coords[times[i]], marker='*', s=100)

print(times[-1], times[0])
ani = FuncAnimation(fig, animate, frames=len(times), interval=2, repeat=True)
plt.show()

'''

TX1EBC_Pow = []
TX1Ustar_Pow = []
TX2_Pow = []
TX3_Pow = []
TX4_Pow = []
TX5_Pow = []
coordinates = []
coordinates_ebc = []
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


        if times[i].astype('datetime64[D]').item().day == 27:
            TX1EBC_Pow.append(PSD_shifted_1[arg_m1])
            coordinates_ebc.append(coords[times[i]])
        else:
            TX1Ustar_Pow.append(PSD_shifted_1[arg_m1])
            coordinates_ustar.append(coords[times[i]])

        TX2_Pow.append(PSD_shifted_2[arg_m2])

        TX3_Pow.append(PSD_shifted_3[arg_m3])

        TX4_Pow.append(PSD_shifted_4[arg_m4])

        TX5_Pow.append(PSD_shifted_5[arg_m5])

        coordinates.append(coords[times[i]])

    '''
    freqs = center + np.fft.fftshift(np.fft.fftfreq(nsamps, 1 / fs))
    idx = find_indices_outside(freqs, cfmin5, cfmax5)

    datum_f = np.fft.fftshift(np.fft.fft(data[times[i]]))

    datum_f[idx] = 0
    p = np.argmax(abs(datum_f))

    #print(10*np.log10(sum(abs(datum_f[np.nonzero(datum_f)])**2)))


    print('F domain, summation of coeffs, no tight filter: \n')
    print(np.sum(abs(datum_f[np.nonzero(datum_f)])))

    print('F domain, summation of coeff squares, no tight filter: \n')
    print(np.sum(abs(datum_f[np.nonzero(datum_f)])**2))

    print('F domain, log of summation of coeff squares, no tight filter: \n')
    print(10*np.log10(np.sum(abs(datum_f[np.nonzero(datum_f)])**2)))


    idx2 = find_indices_outside(freqs, freqs[p-5], freqs[p+5])
    datum_f[idx2] = 0


    print('F domain, summation of coeffs: \n')
    print(np.sum(abs(datum_f)))

    print('F domain, summation of coeff squares: \n')
    print(np.sum(abs(datum_f)**2))

    print('F domain, log of summation of coeff squares: \n')
    print(10*np.log10(np.sum(abs(datum_f)**2)))


    datum_t = np.fft.ifft(np.fft.ifftshift(datum_f))


    print('T domain, summation of coeffs: \n')
    print(np.sum(abs(datum_t)))

    print('T domain, summation of coeff squares: \n')
    print(np.sum(abs(datum_t) ** 2))

    print('T domain, log of summation of coeff squares: \n')
    print(10 * np.log10(np.sum(abs(datum_t) ** 2)))

    #sos = butter(5, [cfmin-center+fs/2, cfmax-center+fs/2], btype='band', fs=0.22e6, output='sos')
    #filtered_signal = sosfilt(sos, datum)
    
    filtered_signal = datum_t
    axs[0].plot(freqs, 10.0 * np.log10(abs( np.fft.fftshift(np.fft.fft(filtered_signal)))))
    axs[0].set_ylim(-40,40)

    axs[1].clear()
    axs[1].scatter(longitude, latitude, marker='.', )
    if times[i] in coords:
        axs[1].scatter(*coords[times[i]], marker='*', s=100)
    '''


for i, time in enumerate(times):
    animate(i)
    if i%1000==0:
        print(i/len(times))


np.save("../files_generated_by_process_data_scripts/TX1EBC_pow_test.npy", TX1EBC_Pow)
np.save("../files_generated_by_process_data_scripts/TX1Ustar_pow_test.npy", TX1Ustar_Pow)
np.save("../files_generated_by_process_data_scripts/TX2_pow_test.npy", TX2_Pow)
np.save("../files_generated_by_process_data_scripts/TX3_pow_test.npy", TX3_Pow)
np.save("../files_generated_by_process_data_scripts/TX4_pow_test.npy", TX4_Pow)
np.save("../files_generated_by_process_data_scripts/TX5_pow_test.npy", TX5_Pow)

np.save("../files_generated_by_process_data_scripts/coordinates_test.npy", coordinates)
np.save("../files_generated_by_process_data_scripts/coordinates_ebc_test.npy", coordinates_ebc)
np.save("../files_generated_by_process_data_scripts/coordinates_ustar_test.npy", coordinates_ustar)
