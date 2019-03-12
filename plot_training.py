import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAGG')
import matplotlib.pyplot as plt
import os


# Copied from baselines.common.plot_util since it wasn't importing correctly
def smooth(y, radius, mode='two_sided', valid_only=False):
    '''
    Smooth signal y, where radius is determines the size of the window

    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]

    valid_only: put nan in entries where the full-sized window is not available

    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2 * radius + 1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius + 1)
        out = np.convolve(y, convkernel, mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel, mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius + 1]
        if valid_only:
            out[:radius] = np.nan
    return out


def load_tensorboard(path: str, property: str):
    files = os.listdir(path)
    files.sort()
    mask = [property in file for file in files]
    values = []
    last_step = 0
    for i, file in enumerate(np.array(files)[mask]):
        data = pd.read_csv(path + file)
        data.sort_values('Step', inplace=True)
        data.Step += last_step
        last_step = data.Step.iloc[-1]
        values.append(data.values)

    return np.vstack(values)


def tensorboard_plot(x, y, path: str, xlabel='Episode', ylabel='Reward',
                     params={'marker': ' ', 'color': 'b', 'linestyle': '-'}):
    fig, ax = plt.subplots(1,1, figsize=(6,4))
    ax.plot(x, y, label='Raw', alpha=0.3, **params)
    smooth_y = smooth(y, radius=10)
    ax.plot(x, smooth_y, label='Smoothed', **params)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    try:
        plt.savefig(path + ylabel, format='png')
    except FileNotFoundError:
        os.makedirs(path)
        plt.savefig(path + ylabel, format='png')
    plt.close(fig)


path = '/home/jsbsim/results/turn_heading/Final_Cessna/'
all_rewards = load_tensorboard(path, 'reward')
all_loss = load_tensorboard(path, 'loss')

tensorboard_plot(all_rewards[:,1], all_rewards[:,-1], path, ylabel='Reward')
tensorboard_plot(all_loss[:,1], all_loss[:,-1], path, ylabel='Loss')