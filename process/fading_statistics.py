import matplotlib.pyplot as plt
import numpy as np
from geopy import distance

def find_unique_lists(list_of_lists):
    unique_lists = set(tuple(lst) for lst in list_of_lists)
    return [list(tpl) for tpl in unique_lists]

tx2_ustar = np.load("../files_generated_by_process_data_scripts/TX1Ustar_pow_stat.npy")
tx3 = np.load("../files_generated_by_process_data_scripts/TX2_pow_stat.npy")
tx4 = np.load("../files_generated_by_process_data_scripts/TX3_pow_stat.npy")
tx5 = np.load("../files_generated_by_process_data_scripts/TX4_pow_stat.npy")
tx6 = np.load("../files_generated_by_process_data_scripts/TX5_pow_stat.npy")

coord = np.load("../files_generated_by_process_data_scripts/coordinates_stat.npy")
coord_ustar = np.load("../files_generated_by_process_data_scripts/coordinates_ustar_stat.npy")


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

for c in coord_ustar:
    dist2.append(distance.distance(tuple(c[::-1]), ustar).m)


coord_unified = np.zeros((len(coord), 2))

skip_list = []

for i, c1 in enumerate(coord):
    if i in skip_list:
        continue
    dummy = []
    dummy_idx = []
    flag = 1
    for j, c2 in enumerate(coord):
        if distance.distance(tuple(c1[::-1]), tuple(c2[::-1])).m < 20:
            flag = 0
            if j not in skip_list:
                dummy.append(list(c2))
                dummy_idx.append(j)
                skip_list.append(j)

    if flag == 0:
        coord_unified[dummy_idx, :] = np.mean(dummy, axis=0)

for i, c1 in enumerate(coord):
    if i not in skip_list:
        coord_unified[i, :] = list(c1)

dist3_ded = []
dist4_ded = []
dist5_ded = []
dist6_ded = []

for c in coord_unified:
    dist3_ded.append(distance.distance(tuple(c[::-1]), guesthousedd).m)
    dist4_ded.append(distance.distance(tuple(c[::-1]), mariodd).m)
    dist5_ded.append(distance.distance(tuple(c[::-1]), morandd).m)
    dist6_ded.append(distance.distance(tuple(c[::-1]), wasatchdd).m)


coord_ustar_unified = np.zeros((len(coord), 2))

skip_list = []

for i, c1 in enumerate(coord):
    if i in skip_list:
        continue
    dummy = []
    dummy_idx = []
    flag = 1
    for j, c2 in enumerate(coord):
        if distance.distance(tuple(c1[::-1]), tuple(c2[::-1])).m < 20:
            flag = 0
            if j not in skip_list:
                dummy.append(list(c2))
                dummy_idx.append(j)
                skip_list.append(j)

    if flag == 0:
        coord_ustar_unified[dummy_idx, :] = np.mean(dummy, axis=0)

for i, c1 in enumerate(coord):
    if i not in skip_list:
        coord_ustar_unified[i, :] = list(c1)

dist2_ded = []


for c in coord_ustar_unified:
    dist2_ded.append(distance.distance(tuple(c[::-1]), ustar).m)


fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)

ax1.scatter(np.log10(dist2_ded), tx2_ustar, s=20, c='blue', alpha=0.7, edgecolors="k")
ax1.grid(visible=True)
ax1.set_xlabel("Log. Distance (log10(d[m]))")
ax1.set_ylabel("10log10(W/Hz)")
ax1.set_title("Ustar \n max. d = %d m" % (max(dist2_ded)))

ax2.scatter(np.log10(dist3_ded), tx3, s=20, c='blue', alpha=0.7, edgecolors="k")
ax2.grid(visible=True)
ax2.set_xlabel("Log. Distance (log10(d[m]))")
ax2.set_title("Guesthouse \n max. d = %d m" % (max(dist3_ded)))

ax3.scatter(np.log10(dist4_ded), tx4, s=20, c='blue',alpha=0.7, edgecolors="k")
ax3.grid(visible=True)
ax3.set_xlabel("Log. Distance (log10(d[m]))")
ax3.set_title("Mario \n max. d = %d m" % (max(dist4_ded)))

ax4.scatter(np.log10(dist5_ded), tx5, s=20, c='blue',alpha=0.7, edgecolors="k")
ax4.grid(visible=True)
ax4.set_xlabel("Log. Distance (log10(d[m]))")
ax4.set_title("Moran \n max. d = %d m" % (max(dist5_ded)))

ax5.scatter(np.log10(dist6_ded), tx6, s=20, c='blue', alpha=0.7, edgecolors="k")
ax5.grid(visible=True)
ax5.set_xlabel("Log. Distance (log10(d[m]))")
ax5.set_title("Wasatch \n max. d = %d m" % (max(dist6_ded)))

plt.suptitle("Power Distribution of Stationary Measurements")

l1=dist3_ded
l2= tx3

l2_chunks = []
current_chunk = []

for i in range(len(l1)):
    if i == 0 or l1[i] == l1[i - 1]:
        current_chunk.append(l2[i])
    else:
        l2_chunks.append(current_chunk)
        current_chunk = [l2[i]]

# Append the last chunk
l2_chunks.append(current_chunk)

print(l2_chunks)

def db_to_linear(dB):
    if isinstance(dB, (int, float)):
        # If a single dB value is provided, convert it to linear
        linear_value = 10 ** (dB / 10)
        return linear_value
    elif isinstance(dB, list):
        # If a list of dB values is provided, convert each one to linear
        linear_values = [np.sqrt(2* (10 ** (x / 10))) for x in dB]
        return linear_values

import seaborn as sns
for i in range(len(l2_chunks)):
    if i<30:
        plt.figure()
        sns.distplot(db_to_linear(l2_chunks[i]), 10)
        plt.show()