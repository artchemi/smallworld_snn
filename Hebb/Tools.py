import numpy as np


class Tools:

    @staticmethod
    def find_index_of_spikes(column, spike_value):
        index = column[column == spike_value].index
        return np.array(index) if not index.empty else np.nan

    @staticmethod
    def find_tau_max(matrix, dt):

        count = 0
        sum_diff = 0
        for col in range(matrix.shape[1]):
            for i in range(len(matrix[:, col])):
                if not np.isnan(matrix[i][col]):
                    for j in range(i + 1, len(matrix[:, col])):
                        if not np.isnan(matrix[j][col]):
                            sum_diff += abs(matrix[i][col] - matrix[j][col])
                            count += 1

        return (sum_diff / count) * 2 * dt
