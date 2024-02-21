import math as mt

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class poisson_filter():

    '''Class for Poisson filtering'''

    @staticmethod
        def spike_generator(image_list, total_time, delay_time, frequency, seed=42):

        '''
        Function generates spike patterns based on the input data array.
        Probability of spiking corresponds to the pixel value.

        Parameters:

        image_list: list of numpy arrays containing input data for each neuron
        total_time (float): total duration for spike generation in seconds
        delay_time (float): time delay between spikes for each neuron in seconds
        frequency (int): frequency of spike generation in Hz
        seed (Optional(int), default = 42): a numerical value that
        generates a new set or repeats pseudo-random numbers

        Returns:

        generated_spikes: numpy array containing spike patterns
        for each neuron over the specified total_time duration
        '''

        np.random.seed(seed)
        generated_spikes = np.zeros((image_list.shape[0], image_list.shape[1], total_time * frequency))

        for z in range(image_list.shape[0]):
            data_norm = (image_list[z] - image_list[z].min()) / (image_list[z].max() - image_list[z].min())
            for i in range(image_list.shape[1]):
                while f <= total_time * frequency:
                    if np.random.choice([0, 1], p=[1 - data_norm[i], data_norm[i]]) == 1:
                        generated_spikes[z][i][f] = 1
                        f = f + mt.ceil(frequency * delay_time)

        return generated_spikes

    @staticmethod
    def raster_plot_spike(spikes, width=10, height=10, cmap='YlGnBu', *args):

        '''
        Generate a raster plot of spikes represented as a numpy array.

        Parameters:

        spikes (np.ndarray): numpy array representing spikes where rows
        correspond to time points and columns correspond to neurons.
        width (Optional(int), default - 10): optional parameter
        for the width of the heatmap plot.
        height (Optional(int), default - 10): optional parameter
        for the height of the heatmap plot.
        cmap (Optional, default - "YlGnBu"): optional parameter for the color map of the heatmap.
        args: additional arguments to be passed to the heatmap function.
        Raises:

        TypeError: If the spikes input is not a numpy array.

        Outputs:

        Heatmap plot displaying the raster plot of
        spikes with time on the horizontal axis and neurons on the vertical axis.'''

        if not isinstance(spikes, np.ndarray):
            raise TypeError

        plt.figure(figsize=(width, height))
        sns.heatmap(spikes, cmap=cmap, *args)
        plt.xlabel('Spikes')
        plt.ylabel('Pixels')
        plt.show()
