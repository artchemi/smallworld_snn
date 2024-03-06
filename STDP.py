import matplotlib.pyplot as plt
import numpy as np
import time
# from neuronpy.graphics import spikeplot


class STDP:
    A_LTP = 5
    A_LTD = 3
    tau_LTP = 1.3
    tau_LTD = 0.75
    tau_bound = 2


def prepare_for_stdp(input_spike_train):
    input_spikes_time = []
    for i in input_spike_train:
        if input_spike_train[i] == 10:
            input_spikes_time.append(i)
    return input_spikes_time


def generate_random_weights(number_of_neurons):
    return np.random.rand(number_of_neurons)



def stdp_event(spike_time_1, spike_time_2, dt):  # multiply by dt
    dw = 0
    delta = (spike_time_1 - spike_time_2) * dt
    if abs(delta) < STDP.tau_bound:
        if delta < 0:
            dw = -STDP.A_LTP * np.exp(delta / STDP.tau_LTP)
        elif delta > 0:
            dw = STDP.A_LTP * np.exp(-delta / STDP.tau_LTP)
    return dw


def generate_pairs():
    pass


def stdp_weight_update(input_spikes_time, spike_train_time, weights):
    sum_constant = sum(weights)
    for n in range(len(weights)):
        for i in range(len(input_spikes_time)):
            for j in range(len(spike_train_time)):
                stdp_event(spike_train_time[j], input_spikes_time[i])
    divisor = sum(weights) / sum_constant
    for i in range(len(weights)):
        weights[i] = weights[i] / divisor

    return weights


def stdp_plot(input_spikes_time, spike_train_time, weights_initial, weights_updated, dw):
    x_init = np.linspace(0,10,10)
    plt.scatter(x_init, dw, color='red')
    plt.ylabel('dw')
    plt.xlabel('neuron')
    plt.show()

    # sp = spikeplot.SpikePlot()
    # sp.plot_spikes(input_spikes_time)


input_spikes_time = np.random.uniform(0,10,10)
input_spikes_time = np.sort(input_spikes_time)
spike_train_time = np.random.uniform(0,10,10)
spike_train_time = np.sort(spike_train_time)
weights = generate_random_weights(10)
weights_init = weights.copy()
print(weights)
weights_updated, deltas = stdp_weight_update(input_spikes_time, spike_train_time, weights)
print(weights_updated)
dw = np.subtract(weights_updated, weights_init)
print(dw)

stdp_plot(input_spikes_time, spike_train_time, weights, weights_updated, dw)
