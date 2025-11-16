import matplotlib.pyplot as plt
import numpy as np
from geopy import distance


tx1_ebc = np.load("./files_generated_by_process_data_scripts/TX1EBC_pow_test.npy")
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


#print(coord)
print(len(coord))
print(len(tx1_ebc))
print(len(tx2_ustar))
print(len(tx3))
print(len(tx4))
print(len(tx5))
print(len(tx6))

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


print(max(dist1))
print(max(dist2))
print(max(dist3))
print(max(dist4))
print(max(dist5))
print(max(dist6))


ind_rem = []
for i in range(len(tx2_ustar)):
    if dist2[i] < 10**2.7 and tx2_ustar[i] < -88:
        ind_rem.append(i)

ind_rem = np.array(ind_rem)

mask = ~np.isin(np.arange(tx2_ustar.size), ind_rem)

# Apply the mask to filter the arrays
tx2_ustar = np.array(tx2_ustar)[mask]
dist2 = np.array(dist2)[mask]

print(len(tx2_ustar))
print(len(dist2))

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
