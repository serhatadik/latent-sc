import matplotlib.pyplot as plt
import numpy as np
from geopy import distance

def find_unique_lists(list_of_lists):
    unique_lists = set(tuple(lst) for lst in list_of_lists)
    return [list(tpl) for tpl in unique_lists]

tx2_ustar = np.load("../files_generated_by_process_data_scripts/TX1Ustar_pow_rot.npy")
tx3 = np.load("../files_generated_by_process_data_scripts/TX2_pow_rot.npy")
tx4 = np.load("../files_generated_by_process_data_scripts/TX3_pow_rot.npy")
tx5 = np.load("../files_generated_by_process_data_scripts/TX4_pow_rot.npy")
tx6 = np.load("../files_generated_by_process_data_scripts/TX5_pow_rot.npy")

coord = np.load("../files_generated_by_process_data_scripts/coordinates_rot.npy")
coord_ustar = np.load("../files_generated_by_process_data_scripts/coordinates_ustar_rot.npy")


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

plt.suptitle("Power Distribution of Rotationary Measurements")



locs = []
var_3 = []
var_4 = []
var_5 = []
var_6 = []

for idx1, uniq_loc in enumerate(find_unique_lists(coord_unified)):
    idxs = np.where(coord_unified==uniq_loc)
    locs.append(uniq_loc)
    var_3.append(np.var(tx3[np.unique(idxs[0])]))
    var_4.append(np.var(tx4[np.unique(idxs[0])]))
    var_5.append(np.var(tx5[np.unique(idxs[0])]))
    var_6.append(np.var(tx6[np.unique(idxs[0])]))


var_2 = []

for idx1, uniq_loc in enumerate(find_unique_lists(coord_ustar_unified)):
    idxs = np.where(coord_ustar_unified==uniq_loc)
    var_2.append(np.var(tx2_ustar[np.unique(idxs[0])]))


dist2_var = []
dist3_var = []
dist4_var = []
dist5_var = []
dist6_var = []

for c in locs:
    dist3_var.append(distance.distance(tuple(c[::-1]), guesthousedd).m)
    dist4_var.append(distance.distance(tuple(c[::-1]), mariodd).m)
    dist5_var.append(distance.distance(tuple(c[::-1]), morandd).m)
    dist6_var.append(distance.distance(tuple(c[::-1]), wasatchdd).m)

for c in locs:
    dist2_var.append(distance.distance(tuple(c[::-1]), ustar).m)

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)

ax1.scatter(np.log10(dist2_var), np.sqrt(var_2), s=20, c='blue', alpha=0.7, edgecolors="k")
ax1.grid(visible=True)
ax1.set_xlabel("Log. Distance (log10(d[m]))")
ax1.set_ylabel("Std. [dB]")
ax1.set_title("Ustar")

ax2.scatter(np.log10(dist3_var), np.sqrt(var_3), s=20, c='blue', alpha=0.7, edgecolors="k")
ax2.grid(visible=True)
ax2.set_xlabel("Log. Distance (log10(d[m]))")
ax2.set_title("Guesthouse")

ax3.scatter(np.log10(dist4_var), np.sqrt(var_4), s=20, c='blue',alpha=0.7, edgecolors="k")
ax3.grid(visible=True)
ax3.set_xlabel("Log. Distance (log10(d[m]))")
ax3.set_title("Mario")

ax4.scatter(np.log10(dist5_var), np.sqrt(var_5), s=20, c='blue',alpha=0.7, edgecolors="k")
ax4.grid(visible=True)
ax4.set_xlabel("Log. Distance (log10(d[m]))")
ax4.set_title("Moran")

ax5.scatter(np.log10(dist6_var), np.sqrt(var_6), s=20, c='blue', alpha=0.7, edgecolors="k")
ax5.grid(visible=True)
ax5.set_xlabel("Log. Distance (log10(d[m]))")
ax5.set_title("Wasatch")

plt.suptitle("Standard Deviation of Rotationary Measurements")

plt.show()
