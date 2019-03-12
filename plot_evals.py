import matplotlib
import numpy as np
import h5py
import scipy.stats as st
import os

matplotlib.use('TkAGG')
import matplotlib.pyplot as plt


class AgentEval:
    def __init__(self, actions, states):
        self.good_eps, self.bad_eps, self.num_eps = self.eps_info(states)
        self.good_action_hist = [actions[:, ep[0]:ep[0] + ep[1]] for ep in self.good_eps]
        self.good_action_stats = [st.describe(self.good_action_hist[i], axis=1) for i, ep in enumerate(self.good_eps)]
        self.good_states_init = [states[:, ep[0]] for ep in self.good_eps]
        self.good_errors = [states[[-2, -4], ep[0]:ep[0] + ep[1]] for ep in self.good_eps]
        self.bad_errors = [states[[-2, -4], ep[0]:ep[0] + ep[1]] for ep in self.bad_eps]
        self.bad_action_hist = [actions[:, ep[0]:ep[0] + ep[1]] for ep in self.bad_eps]
        self.bad_action_stats = [st.describe(self.bad_action_hist[i], axis=1) for i, ep in enumerate(self.bad_eps)]
        self.bad_states_init = [states[:, ep[0]] for ep in self.bad_eps]

    def eps_info(self, states):
        ep_starts = np.where(states[-1, :] == 299.0)[0]
        ep_len = np.diff(ep_starts)
        ep_len = np.append(ep_len, states.shape[1] - ep_starts[-1])

        good_starts = []
        bad_starts = []
        good_len = []
        bad_len = []
        for i, ep in enumerate(ep_len):
            if ep == 299:
                good_starts.append(ep_starts[i])
                good_len.append(ep)
            else:
                bad_starts.append(ep_starts[i])
                bad_len.append(ep)

            good_eps = list(zip(good_starts, good_len))
            bad_eps = list(zip(bad_starts, bad_len))

        return good_eps, bad_eps, ep_starts.shape[0]


def load_histories(filepath: str, name: str = 'act', task: str = 'steadyheading', aircraft: str = 'cessna172',
                   models: list = None):
    with h5py.File(filepath, mode='r') as file:
        histories = {}
        if models is None:
            models = list(file['/' + task + '/' + aircraft].keys())
        for model in models:
            data = file['/' + task + '/' + aircraft + '/' + model + '/' + name]
            histories[model] = data[...]

        return histories


def action_plot(agent_eval: AgentEval, model: str, alignment: str='good', path='/home/jsbsim/results/', task='steadyheading', agent_num=1):
    if alignment.lower() == 'good':
        history = agent_eval.good_action_hist
    elif alignment.lower() == 'bad' or 'evil':
        history = agent_eval.bad_action_hist
    else:
        ValueError('This alignment is not recognized')
    for n, hist in enumerate(history):
        t = np.arange(0, hist.shape[1]/5, 0.2)
        fig, axs = plt.subplots(3, 1, figsize=(6,5))
        axs[0].plot(t, hist[0, :])
        axs[0].set_ylabel('Norm Ail Cmd')
        axs[0].set_ylim([-1.1, 1.1])
        axs[0].set_yticks(np.arange(-1, 1.5, 0.5))
        axs[0].set_yticks(np.arange(-0.75,1.25,0.5),minor=True)
        axs[0].set_yticklabels([-1.0, -0.5, 0.0, 0.5, 1.0])
        axs[0].grid(True, linestyle=':')
        axs[1].plot(t, hist[1, :])
        axs[1].set_ylabel('Norm Ele Cmd')
        axs[1].set_yticks(np.arange(-1, 1.5, 0.5))
        axs[1].set_yticks(np.arange(-0.75, 1.25, 0.5), minor=True)
        axs[1].set_yticklabels([-1.0, -0.5, 0.0, 0.5, 1.0])
        axs[1].set_ylim([-1.1, 1.1])
        axs[1].grid(True, linestyle=':')
        axs[2].plot(t, hist[2, :])
        axs[2].set_ylabel('Norm Rdr Cmd')
        axs[2].set_yticks(np.arange(-1, 1.5, 0.5))
        axs[2].set_yticks(np.arange(-0.75, 1.25, 0.5), minor=True)
        axs[2].set_yticklabels([-1.0, -0.5, 0.0, 0.5, 1.0])
        axs[2].set_ylim([-1.1, 1.1])
        axs[2].set_xlabel('Episode Time (s)')
        axs[2].grid(True, linestyle=':')
        plt.tight_layout()


        savepath = path+'plots/' + task + '/'+model + '/' + str(agent_num) + '/'
        try:
            plt.savefig(savepath + 'ep_actions_' + alignment + str(n), format='png')
        except FileNotFoundError:
            os.makedirs(savepath)
            plt.savefig(savepath + 'ep_actions_' + alignment + str(n), format='png')
        plt.close(fig)


def error_plot(agent_eval: AgentEval, model: str, alignment: str='good', path='/home/jsbsim/results/', task='steadyheading', agent_num=1):
    if alignment.lower() == 'good':
        history = agent_eval.good_errors
    elif alignment.lower() == 'bad' or 'evil':
        history = agent_eval.bad_errors
    else:
        ValueError('This alignment is not recognized')
    for n, hist in enumerate(history):
        t = np.arange(0, hist.shape[1]/5, 0.2)
        fig, axs = plt.subplots(2, 1, figsize=(6,5))
        axs[0].plot(t, hist[0, :])
        axs[0].set_ylabel('Track Error (deg)')
        axs[0].set_ylim([-180, 180])
        axs[0].set_yticks(np.arange(-180,270,90))
        axs[0].set_yticklabels(np.arange(-180,270,90))
        axs[0].grid(True, linestyle=':')
        axs[1].plot(t, hist[1, :])
        axs[1].set_ylabel('Altitude Error (ft)')
        axs[1].set_xlabel('Episode Time (s)')
        axs[1].set_ylim([-1000, 1000])
        axs[1].grid(True, linestyle=':')
        plt.tight_layout()

        savepath = path + 'plots/' + task + '/' + model + '/' + str(agent_num) + '/'
        try:
            plt.savefig(savepath + 'ep_errors_' + alignment + str(n), format='png')
        except FileNotFoundError:
            os.makedirs(savepath)
            plt.savefig(savepath + 'ep_errors_' + alignment + str(n), format='png')
        plt.close(fig)


def error_histogram(agent_eval: AgentEval, model: str, alignment: str='good', path='/home/jsbsim/results/', task='steadyheading', agent_num=1):
    if alignment.lower() == 'good':
        history = agent_eval.good_errors
    elif alignment.lower() == 'bad' or 'evil':
        history = agent_eval.bad_errors
    else:
        ValueError('This alignment is not recognized')

    all_errors = np.hstack(history)

    num_bins = 20
    fig, axs = plt.subplots(1, 2, figsize=(6,3.75))
    axs[0].hist(all_errors[0], num_bins, range=[-180,180])
    axs[0].set_xlabel('Track Error (deg)')
    axs[0].set_xticks(np.arange(-180, 270, 90))
    axs[0].set_xticklabels(np.arange(-180, 270, 90))
    axs[1].hist(all_errors[1], num_bins, range=[-1000,1000])
    axs[1].set_xlabel('Altitude Error (ft)')
    plt.tight_layout()

    savepath = path + 'plots/' + task + '/' + model + '/' + str(agent_num) + '/'
    try:
        plt.savefig(savepath + 'histogram_' + alignment, format='png')
    except FileNotFoundError:
        os.makedirs(savepath)
        plt.savefig(savepath + 'histogram_' + alignment, format='png')
    plt.close(fig)


def action_histogram(agent_eval: AgentEval, model: str, alignment: str='good', path='/home/jsbsim/results/', task='steadyheading', agent_num=1):
    if alignment.lower() == 'good':
        history = agent_eval.good_action_hist
    elif alignment.lower() == 'bad' or 'evil':
        history = agent_eval.bad_action_hist
    else:
        ValueError('This alignment is not recognized')

    all_actions = np.hstack(history)

    num_bins = 20
    fig, axs = plt.subplots(1, 3, figsize=(6,3.75))
    axs[0].hist(all_actions[0], num_bins, range=[-1,1])
    axs[0].set_xlabel('Norm Ail Cmd')
    axs[1].hist(all_actions[1], num_bins, range=[-1,1])
    axs[1].set_xlabel('Norm Ele Cmd')
    axs[2].hist(all_actions[2], num_bins, range=[-1,1])
    axs[2].set_xlabel('Norm Rud Cmd')
    plt.tight_layout()

    savepath = path + 'plots/' + task + '/' + model + '/' + str(agent_num) + '/'
    try:
        plt.savefig(savepath + 'act_histogram_' + alignment, format='png')
    except FileNotFoundError:
        os.makedirs(savepath)
        plt.savefig(savepath + 'act_histogram_' + alignment, format='png')
    plt.close(fig)


# def plot_initial_state()



path = '/home/jsbsim/results/'
file = 'state_action_history.h5'
filepath = path + file
models = ['ppo2_rnn_h128_mb4_g95_thr90_lr1e-4_2'] #'ppo2_fg_turnheading_rnn_h128_mb4_g95_thr90_lr1e-4'
task = 'steadyheading'

act_history = load_histories(filepath, name='act', models=models, task=task, aircraft='F15')
state_history = load_histories(filepath, name='obs', models=models, task=task, aircraft='F15')

for key in act_history.keys():
    if act_history[key].ndim > 2:
        for net in range(act_history[key].shape[0]):
            net_eval = AgentEval(act_history[key][net, ...], state_history[key][net, ...])
            action_plot(net_eval, key, alignment='good', path=path, task=task, agent_num=net)
            action_plot(net_eval, key, alignment='bad', path=path, task=task, agent_num=net)
            error_plot(net_eval, key, alignment='good', path=path, task=task, agent_num=net)
            error_plot(net_eval, key, alignment='bad', path=path, task=task, agent_num=net)
            error_histogram(net_eval, key, alignment='good', path=path, task=task, agent_num=net)
            error_histogram(net_eval, key, alignment='bad', path=path, task=task, agent_num=net)
            action_histogram(net_eval, key, alignment='good', path=path, task=task, agent_num=net)
            action_histogram(net_eval, key, alignment='bad', path=path, task=task, agent_num=net)





