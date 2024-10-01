from brian2 import *
from tqdm import tqdm
import numpy as np


# n_input = 28 * 28  # input layer
n_e = 100  # e - excitatory
n_i = n_e  # i - inhibitory

v_rest_e = -60. * mV  # v - membrane potential
v_reset_e = -65. * mV
v_thresh_e = -52. * mV

v_rest_i = -60. * mV
v_reset_i = -45. * mV
v_thresh_i = -40. * mV

taupre = 20 * ms
taupost = taupre
gmax = .05  # .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

# Apre and Apost - presynaptic and postsynaptic traces, lr - learning rate
stdp = '''w : 1
    lr : 1 (shared)
    dApre/dt = -Apre / taupre : 1 (event-driven)
    dApost/dt = -Apost / taupost : 1 (event-driven)'''
pre = '''ge += w
    Apre += dApre
    w = clip(w + lr*Apost, 0, gmax)'''
post = '''Apost += dApost
    w = clip(w + lr*Apre, 0, gmax)'''


class Model:
    """
    Модель из оригинальной статьи без малого мира
    """
    def __init__(self, n_input: int, n_hidden: int, dir_name: str, lr: float, save_flag=False) -> None:
        app = {}

        self.n_input = n_input
        self.n_hid = n_hidden

        self.dir_name = dir_name

        self.save_flag = save_flag
        self.learning_rate = lr

        # Входные изображения кодируются как скорость Пуассоновских генераторов
        app['PG'] = PoissonGroup(self.n_input, rates=np.zeros(self.n_input) * Hz, name='PG')

        # Группа возбуждающих нейронов
        neuron_e = '''
            dv/dt = (ge*(0*mV-v) + gi*(-100*mV-v) + (v_rest_e-v)) / (100*ms) : volt
            dge/dt = -ge / (5*ms) : 1
            dgi/dt = -gi / (10*ms) : 1
            '''
        app['EG'] = NeuronGroup(self.n_hid, neuron_e, threshold='v>v_thresh_e', refractory=5 * ms, reset='v=v_reset_e',
                                method='euler', name='EG')
        app['EG'].v = v_rest_e - 20. * mV

        # if (debug):
        #     app['ESP'] = SpikeMonitor(app['EG'], name='ESP')
        #     app['ESM'] = StateMonitor(app['EG'], ['v'], record=True, name='ESM')
        #     app['ERM'] = PopulationRateMonitor(app['EG'], name='ERM')

        # Группа ингибирующих нейронов
        neuron_i = '''
            dv/dt = (ge*(0*mV-v) + (v_rest_i-v)) / (10*ms) : volt
            dge/dt = -ge / (5*ms) : 1
            '''
        app['IG'] = NeuronGroup(self.n_hid, neuron_i, threshold='v>v_thresh_i', refractory=2 * ms, reset='v=v_reset_i',
                                method='euler', name='IG')
        app['IG'].v = v_rest_i - 20. * mV

        
        # app['ISP'] = SpikeMonitor(app['IG'], name='ISP')
        # app['ISM'] = StateMonitor(app['IG'], ['v'], record=True, name='ISM')
        # app['IRM'] = PopulationRateMonitor(app['IG'], name='IRM')

        # poisson generators one-to-all excitatory neurons with plastic connections
        app['S1'] = Synapses(app['PG'], app['EG'], stdp, on_pre=pre, on_post=post, method='euler', name='S1')
        app['S1'].connect()
        app['S1'].w = 'rand()*gmax'  # random weights initialisation
        app['S1'].lr = 1  # enable stdp

        # excitation neurons one-to-one inhibitory neurons
        app['S2'] = Synapses(app['EG'], app['IG'], 'w : 1', on_pre='ge += w', name='S2')
        app['S2'].connect(j='i')
        app['S2'].delay = 'rand()*10*ms'
        app['S2'].w = 3  # very strong fixed weights to ensure corresponding inhibitory neuron will always fire

        # inhibitory neurons one-to-all-except-one excitatory neurons
        app['S3'] = Synapses(app['IG'], app['EG'], 'w : 1', on_pre='gi += w', name='S3')
        app['S3'].connect(condition='i!=j')
        app['S3'].delay = 'rand()*5*ms'
        app['S3'].w = .03  # weights are selected in such a way as to maintain a balance between excitation and ibhibition

        self.net = Network(app.values())
        self.net.run(0 * second)

    def __getitem__(self, key):
        return self.net[key]

    def train(self, X, epoch=1, weight_save=False):
        """Функция для обучения 

        Args:
            X (_type_): Входные данные MNIST 
            epoch (int, optional): Количество эпох. Defaults to 1.
            weight_save (bool, optional): Флаг для сохранения межслойных весов. Defaults to False.
        """
        self.net['S1'].lr = self.learning_rate  # stdp on

        for ep in range(epoch):
            count = 0
            for idx in tqdm(range(len(X))):
                # active mode
                self.net['PG'].rates = X[idx].ravel() * Hz
                self.net.run(0.35 * second)

                # passive mode
                self.net['PG'].rates = np.zeros(self.n_input) * Hz
                self.net.run(0.15 * second)

                # сохранение межслойных весов
                if weight_save:
                    with open(f'{self.dir_name}/weights/weight_data{count}.npy', 'wb') as file:
                        np.save(file, np.array(self.net['S1'].w).reshape((784, self.n_hid)))

                count += 1
                
            if self.save_flag == True:
                self.net.store('train', f'{self.dir_name}/epoch_{ep+1}/chk_{ep+1}.b2')

    def evaluate(self, X, chk: str):
        self.net['S1'].lr = 0  # stdp off

        if chk != None:
            self.net.restore(name='train', filename=chk)

        
            # rate monitor to count spikes
        mon = SpikeMonitor(self.net['IG'], name='RM')
        mem_mon = StateMonitor(self.net['IG'], 'v', record=True, name='ISM')

        self.net.add(mon)
        self.net.add(mem_mon)

        # active mode
        self.net['PG'].rates = X.ravel() * Hz
        self.net.run(0.35 * second)

        # spikes per neuron foreach image
        features = np.array(mon.count, dtype=int32)
            
        # passive mode
        self.net['PG'].rates = np.zeros(self.n_input) * Hz
        self.net.run(0.15 * second)

        mem_features = mem_mon.v

        self.net.remove(self.net['RM'])
        self.net.remove(self.net['ISM'])

        features = np.array(features)
        features = features.reshape((10, 10))

        return features, mem_features
