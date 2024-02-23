import numpy as np

class weight_calculation():

    @staticmethod
    def intralayer_hebbian(t_1, t_2, t_max, sigma_max):

        """
        Функция intralayer_hebbian расчитывает весовой коэффициент между двумя нейронами в слое с помощью правила Хэбба.

        Params:

        t_1 (float): Время активации первого нейрона.
        t_2 (float): Время активации второго нейрона.
        t_max (float): Максимальное время активации.
        sigma_max (float): Максимальное значение весового коэффициента.

        Returns:
        calculation (float): Значение весового коэффициента, рассчитанное по
        формуле Хэбба и ограниченное в диапазоне [-sigma_max, sigma_max].
        """

        if t_1 - t_2 >= t_max:
            raise ValueError('Spikes time difference cannot be more or equal to the t_max. '
                             'Increase t_max parameter')

        calculation = (- 0.5 * np.log((t_1 - t_2) / (t_max - (t_1 - t_2)))) * sigma_max / 3.57
        return np.clip(calculation, a_min=-sigma_max, a_max=sigma_max)
