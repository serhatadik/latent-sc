import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

'''
def plot_graphs(env_map=None, tirem_rssi=None):
    """Plots tirem RSSI values. If None, load values."""
    # env_map = np.load('tx_powder.npz', allow_pickle=True)['map']
    # tirem_rssi = np.load('tirem_rssi.npy')
    plotter(env_map)
    plt.title('TIREM RSSI Prediction')
    plotter(tirem_rssi)
    contour_plot(tirem_rssi)


def contour_plot(tirem_rssi):
    lvls = np.linspace(-200, -40, 10)
    x = np.arange(tirem_rssi.shape[0])
    y = np.arange(tirem_rssi.shape[1])
    plt.contourf(x, y, tirem_rssi, levels=lvls, cmap='hot')
    plt.axis('scaled')
    plt.grid(which='minor')
    plt.colorbar()
    plt.show()
'''

def plotter(value_map, title):
    plt.imshow(value_map[1:value_map.shape[0] - 1, 1:value_map.shape[1] - 1], origin='lower')
    plt.axis('scaled')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("UTM_E [m]")
    plt.ylabel("UTM_N [m]")
    plt.show()
