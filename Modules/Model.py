from brian2 import *
import numpy as np
import networkx as nx
import tqdm


class SNN_model():

    '''Этот класс реализует модель спайковой нейронной сети (SNN) с использованием библиотеки Brian2.
    Модель включает в себя возбудительные и тормозные нейроны,
    а также синапсы с пластичностью, основанной на временной зависимости спайков (STDP)'''

    def __init__(self, neuron_exc, neuron_inh, n_input = 28 * 28, n_exс=100, n_inh=100,
                 v_rest_e=-60., v_reset_e=-65.,
                 v_thresh_e=-52., v_rest_i= -60.,
                 v_reset_i=-45., v_thresh_i=-40.,
                 taupre= 20, taupost= 20,
                 gmax= .3, gmax_small= .3, dApre= .01,
                 debug=False, small_world=False):

        '''
        params:
        neuron_exc(str): Уравнения для возбудительных нейронов.
        neuron_inh(str): Уравнения для тормозных нейронов.
        n_input(int): Количество входных нейронов (по умолчанию 28*28).
        n_exс(int): Количество возбудительных нейронов (по умолчанию 100).
        n_inh(int): Количество тормозных нейронов (по умолчанию 100).
        v_rest_e(float): Потенциал покоя для возбудительных нейронов.
        v_reset_e(float): Потенциал сброса для возбудительных нейронов.
        v_thresh_e(float): Пороговый потенциал для возбудительных нейронов.
        v_rest_i(float): Потенциал покоя для тормозных нейронов.
        v_reset_i(float): Потенциал сброса для тормозных нейронов.
        v_thresh_i(float): Пороговый потенциал для тормозных нейронов.
        taupre(float): Время затухания потенциала до спайка.
        taupost(float): Время затухания потенциала после спайка.
        gmax(float): Максимальная проводимость синапсов.
        gmax_small(float): Максимальная проводимость для маломировых синапсов.
        dApre: Изменение потенциала до спайка.
        debug: Флаг для включения режима отладки.
        small_world: Флаг для включения малого мира в сеть.
        '''


        self.neuron_exc=neuron_exc
        self.neuron_inh=neuron_inh
        self.n_input = n_input
        self.n_exc = n_exс
        self.n_inh = n_inh
        self.v_rest_e = v_rest_e * mV
        self.v_reset_e = v_reset_e * mV
        self.v_thresh_e = v_thresh_e * mV
        self.v_rest_i = v_rest_i * mV
        self.v_reset_i = v_reset_i * mV
        self.v_thresh_i = v_thresh_i * mV
        self.taupre = taupre * ms
        self.taupost = taupost * ms
        self.gmax = gmax
        self.gmax_small = gmax_small
        self.dApre = dApre * self.gmax
        self.dApost = self.dApre * self.taupre / self.taupost * 1.05 *self.gmax
        self.small_world = small_world
        self.debug = debug

        stdp = '''w : 1
            lr : 1 (shared)
            dApre/dt = -Apre / taupre : 1 (event-driven)
            dApost/dt = -Apost / taupost : 1 (event-driven)'''
        pre = '''ge += w
            Apre += dApre
            w = clip(w + lr*Apost, 0, gmax)'''
        post = '''Apost += dApost
            w = clip(w + lr*Apre, 0, gmax)'''


        app = {}

        if small_world:
            self.g = nx.watts_strogatz_graph(self.n_exc, 6, 0.5)

        # input images as rate encoded Poisson generators
        app['PG'] = PoissonGroup(self.n_input, rates=np.zeros(self.n_input) * Hz, name='PG')

        # excitatory group
        app['EG'] = NeuronGroup(self.n_exc, self.neuron_exc, threshold='v>v_thresh_e', refractory=5 * ms, reset='v=v_reset_e',
                                method='euler', name='EG')
        app['EG'].v = self.v_rest_e - 20. * mV

        if (debug):
            app['ESP'] = SpikeMonitor(app['EG'], name='ESP')
            app['ESM'] = StateMonitor(app['EG'], ['v'], record=True, name='ESM')
            app['ERM'] = PopulationRateMonitor(app['EG'], name='ERM')

        # ibhibitory group

        app['IG'] = NeuronGroup(self.n_inh, self.neuron_i, threshold='v>v_thresh_i', refractory=2 * ms, reset='v=v_reset_i',
                                method='euler', name='IG')
        app['IG'].v = self.v_rest_i - 20. * mV

        if (self.debug):
            app['ISP'] = SpikeMonitor(app['IG'], name='ISP')
            app['ISM'] = StateMonitor(app['IG'], ['v'], record=True, name='ISM')
            app['IRM'] = PopulationRateMonitor(app['IG'], name='IRM')

        # poisson generators one-to-all excitatory neurons with plastic connections
        app['S1'] = Synapses(app['PG'], app['EG'], stdp, on_pre=pre, on_post=post, method='euler', name='S1')
        app['S1'].connect()
        app['S1'].w = 'rand()*gmax'  # random weights initialisation
        app['S1'].lr = 1  # enable stdp

        if small_world:
            app['S_small_world'] = Synapses(app['EG'], app['EG'], stdp, on_pre=pre, on_post=post, method='euler',
                                            name='S_small_world')

            for (u, v) in self.g.edges():
                print(f'{u}---{v}')
                app['S_small_world'].connect(j=u, i=v)
                app['S_small_world'].w = 'rand()*gmax_small'  # здесь нужно сделать так, чтобы генерировались
                # рандомные веса на порядок меньше, чем основные
            app['S_small_world'].lr = 1

            self.res_0 = [0]
            for g, v in self.g.edges():
                if g == 0:
                    self.res_0.append(v)

        if (self.debug):
            # some synapses
            app['S1M'] = StateMonitor(app['S1'], ['w', 'Apre', 'Apost'], record=app['S1'][380, :4], name='S1M')

            # excitatory neurons one-to-one inhibitory neurons
        app['S2'] = Synapses(app['EG'], app['IG'], 'w : 1', on_pre='ge += w', name='S2')
        app['S2'].connect(j='i')
        app['S2'].delay = 'rand()*10*ms'
        app['S2'].w = 3  # very strong fixed weights to ensure corresponding inhibitory neuron will always fire

        # inhibitory neurons one-to-all-except-one excitatory neurons
        app['S3'] = Synapses(app['IG'], app['EG'], 'w : 1', on_pre='gi += w', name='S3')
        app['S3'].connect(condition='i!=j')
        app['S3'].delay = 'rand()*5*ms'
        app[
            'S3'].w = .03  # weights are selected in such a way as to maintain a balance between excitation and ibhibition

        self.net = Network(app.values())
        self.net.run(0 * second)

    def __getitem__(self, key):
        return self.net[key]

    def train(self, X, epoch=1):

        self.net['S1'].lr = 1
        self.net['S_small_world'].lr = 1  # stdp on

        for ep in range(epoch):
            for idx in tqdm(range(len(X))):

                mon = SpikeMonitor(self.net['EG'], name='TestRM')
                mon2 = SpikeMonitor(self.net['IG'], name='TestIM')

                self.net.add(mon)
                self.net.add(mon2)

                # active mode
                self.net['PG'].rates = X[idx].ravel() * Hz
                self.net.run(0.35 * second)

                # passive mode
                self.net['PG'].rates = np.zeros(self.n_input) * Hz
                self.net.run(0.15 * second)

                self.net.remove(self.net['TestRM'])
                self.net.remove(self.net['TestIM'])

    def evaluate(self, X):
        self.net['S1'].lr = 0
        self.net['S_small_world'].lr = 0  # stdp off

        features = []
        for idx in tqdm(range(len(X))):
            # rate monitor to count spikes

            mon = SpikeMonitor(self.net['EG'], name='RM')

            self.net.add(mon)

            # active mode
            self.net['PG'].rates = X[idx].ravel() * Hz
            self.net.run(0.35 * second)

            # spikes per neuron foreach image
            features.append(np.array(mon.count, dtype=int8))

            # passive mode
            self.net['PG'].rates = np.zeros(self.n_input) * Hz
            self.net.run(0.15 * second)

            self.net.remove(self.net['RM'])

        return features

