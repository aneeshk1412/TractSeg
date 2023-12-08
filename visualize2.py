import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--filepath1", type=str, required=True)
parser.add_argument("--filepath2", type=str, required=True)
parser.add_argument("--filepath3", type=str, required=True)
parser.add_argument("--channel", type=int, default=17)
parser.add_argument("--axis", type=str, default='z', choices=['x', 'y', 'z'])

args = parser.parse_args()

def load_data(filepath):
    data = np.load(filepath)
    _, d, x, y, z = data.shape
    data = data.reshape(d, x, y, z)
    data = data.transpose(1, 2, 3, 0)
    return data

# Load data from each filepath
data1 = load_data(args.filepath1)[:, :, :, args.channel]
data2 = load_data(args.filepath2)[:, :, :, args.channel]
data3 = load_data(args.filepath3)[:, :, :, args.channel]

def transpose_volume(data, axis):
    if axis == 'z':
        return data.transpose(2, 1, 0)
    elif axis == 'y':
        return data.transpose(1, 0, 2)
    elif axis == 'x':
        return data.transpose(0, 2, 1)

volume1 = transpose_volume(data1, args.axis)
volume2 = transpose_volume(data2, args.axis)
volume3 = transpose_volume(data3, args.axis)

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith("keymap."):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def process_key(event):
    fig = event.canvas.figure
    if event.key == "j":
        change_slice(fig, -1)
    elif event.key == "k":
        change_slice(fig, 1)
    fig.canvas.draw()

def change_slice(fig, delta):
    for ax in fig.axes:
        previous_index = ax.index
        volume = ax.volume
        ax.index = (previous_index + delta) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])

def multi_slice_viewer(volumes):
    remove_keymap_conflicts({"j", "k"})
    fig, axes = plt.subplots(1, 3)
    for ax, volume in zip(axes, volumes):
        ax.volume = volume
        ax.index = volume.shape[0] // 2
        ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect("key_press_event", process_key)
    plt.show()

multi_slice_viewer([volume1, volume2, volume3])
