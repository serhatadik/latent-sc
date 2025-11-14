import matplotlib.pyplot as plt
import numpy as np
from geopy import distance


tx1_ebc = np.load("files_generated_by_process_data_scripts/TX1EBC_pow_test.npy")
tx2 = np.load("files_generated_by_process_data_scripts/TX2_pow_test.npy")
tx3 = np.load("files_generated_by_process_data_scripts/TX3_pow_test.npy")
tx4 = np.load("files_generated_by_process_data_scripts/TX4_pow_test.npy")
tx5 = np.load("files_generated_by_process_data_scripts/TX5_pow_test.npy")

tx2 = tx2[:, np.newaxis]
tx3 = tx3[:, np.newaxis]
tx4 = tx4[:, np.newaxis]
tx5 = tx5[:, np.newaxis]

coord = np.load("files_generated_by_process_data_scripts/coordinates_test.npy")
coord_ebc = np.load("files_generated_by_process_data_scripts/coordinates_ebc_test.npy")
print(tx2.shape)
print(tx2[0])
print(coord.shape)
print(coord[0, :])
ebcdd = (40.76702, -111.83807)
guesthousedd = (40.76749, -111.83607)
mariodd = (40.77283, -111.84088)
morandd = (40.76989, -111.83869)
wasatchdd = (40.77108, -111.84316)


x = coord.shape[0]
y = coord_ebc.shape[0]
pad_length = x - y


if tx1_ebc.ndim == 1:
    tx1_ebc_padded = np.pad(tx1_ebc, (0, pad_length), 'constant', constant_values=0)
elif tx1_ebc.ndim == 2 and tx1_ebc.shape[1] == 1:
    tx1_ebc_padded = np.pad(tx1_ebc, ((0, pad_length), (0, 0)), 'constant', constant_values=0)

tx1_ebc_padded = tx1_ebc_padded[:, np.newaxis]
print(tx1_ebc.shape)
print(tx1_ebc_padded.shape)
print(tx2.shape)
coord_ebc_padded = np.pad(coord_ebc, ((0, pad_length), (0, 0)), 'constant', constant_values=0)
print(coord_ebc_padded.shape)
print(coord.shape)
combined_array = np.hstack([
    tx1_ebc_padded, coord_ebc_padded[:, 1:2], coord_ebc_padded[:, 0:1],
    tx2, coord[:, 1:2], coord[:, 0:1],
    tx3, coord[:, 1:2], coord[:, 0:1],
    tx4, coord[:, 1:2], coord[:, 0:1],
    tx5, coord[:, 1:2], coord[:, 0:1]
])

print(combined_array.shape)
print(combined_array[:10, :6])

from scipy.io import savemat

# Dictionary containing the combined array
mat_data = {'measurements': combined_array}

# Saving to a .mat file
savemat('dd_meas_data.mat', mat_data)

'''

#print(coord)
print(len(coord))
print(len(tx1_ebc))
print(len(tx3))
print(len(tx4))
print(len(tx5))
print(len(tx6))

for c in coord:
    dist3.append(distance.distance(tuple(c[::-1]), guesthousedd).m)
    dist4.append(distance.distance(tuple(c[::-1]), mariodd).m)
    dist5.append(distance.distance(tuple(c[::-1]), morandd).m)
    dist6.append(distance.distance(tuple(c[::-1]), wasatchdd).m)

for c in coord_ebc:
    dist1.append(distance.distance(tuple(c[::-1]), ebcdd).m)



fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1,6)

print(len(np.log10(dist1)))
print(len(tx1_ebc))
ax1.scatter(np.log10(dist1), tx1_ebc, s=20, c='blue', alpha=0.7, edgecolors="k")
ax1.grid(visible=True, color='green', linestyle='--', which='major')
ax1.set_xlabel("Log. Distance (log10(d[m]))")
ax1.set_ylabel("10log10(W/Hz)")
(m, b) = np.polyfit(np.log10(dist1), tx1_ebc, 1)
ax1.plot(np.log10(dist1), m*np.log10(dist1)+b, linewidth=4, color='red', linestyle='-.')
ax1.set_title("EBC \n max. d = %d m \n n = %f" % (max(dist1), abs(m/10)) )

ax2.scatter(np.log10(dist2), tx2_ustar, s=20, c='blue', alpha=0.7, edgecolors="k")
#ax4.set_xscale("log")
ax2.grid(visible=True, color='green', linestyle='--', which='major')
ax2.set_xlabel("Log. Distance (log10(d[m]))")
(m, b) = np.polyfit(np.log10(dist2), tx2_ustar, 1)
ax2.plot(np.log10(dist2), m*np.log10(dist2)+b, linewidth=4, color='red', linestyle='-.')
ax2.set_title("Ustar \n max. d = %d m \n n = %f" % (max(dist2), abs(m/10)) )


ax3.scatter(np.log10(dist3), tx3, s=20, c='blue', alpha=0.7, edgecolors="k")
#ax1.set_xscale("log")
ax3.grid(visible=True, color='green', linestyle='--', which='major')
ax3.set_xlabel("Log. Distance (log10(d[m]))")
(m, b) = np.polyfit(np.log10(dist3), tx3, 1)
ax3.plot(np.log10(dist3), m*np.log10(dist3)+b, linewidth=4, color='red', linestyle='-.')
ax3.set_title("Guesthouse \n max. d = %d m \n n = %f" % (max(dist3), abs(m/10)) )


ax4.scatter(np.log10(dist4), tx4, s=20, c='blue',alpha=0.7, edgecolors="k")
#ax2.set_xscale("log")
ax4.grid(visible=True, color='green', linestyle='--', which='major')
ax4.set_xlabel("Log. Distance (log10(d[m]))")
(m, b) = np.polyfit(np.log10(dist4), tx4, 1)
ax4.plot(np.log10(dist4), m*np.log10(dist4)+b, linewidth=4, color='red', linestyle='-.')
ax4.set_title("Mario \n max. d = %d m \n n = %f" % (max(dist4), abs(m/10)) )

ax5.scatter(np.log10(dist5), tx5, s=20, c='blue',alpha=0.7, edgecolors="k")
#ax3.set_xscale("log")
ax5.grid(visible=True, color='green', linestyle='--', which='major')
ax5.set_xlabel("Log. Distance (log10(d[m]))")
(m, b) = np.polyfit(np.log10(dist5), tx5, 1)
ax5.plot(np.log10(dist5), m*np.log10(dist5)+b, linewidth=4, color='red', linestyle='-.')
ax5.set_title("Moran \n max. d = %d m \n n = %f" % (max(dist5), abs(m/10)) )

ax6.scatter(np.log10(dist6), tx6, s=20, c='blue', alpha=0.7, edgecolors="k")
#ax4.set_xscale("log")
ax6.grid(visible=True, color='green', linestyle='--', which='major')
ax6.set_xlabel("Log. Distance (log10(d[m]))")
(m, b) = np.polyfit(np.log10(dist6), tx6, 1)
ax6.plot(np.log10(dist6), m*np.log10(dist6)+b, linewidth=4, color='red', linestyle='-.')
ax6.set_title("Wasatch \n max. d = %d m \n n = %f" % (max(dist6), abs(m/10)) )

plt.suptitle("Power Distribution of Mobile Measurements")


plt.show()
'''