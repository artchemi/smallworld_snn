import numpy as np

class Tools:

    @staticmethod
    def find_index_of_spikes(column, spike_value):
        index = column[column == spike_value].index
        return np.array(index) if not index.empty else None

    @staticmethod
    def find_tau_max(matrix, dt):
        matrix = np.where(matrix == None, np.nan, matrix)
        matrix = matrix.astype(float)
        max_diff = np.nanmax(matrix, axis=0) - np.nanmin(matrix, axis=0)
        return max(max_diff) * 2 * dt