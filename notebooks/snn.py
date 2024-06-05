import brian2 as b2
import matplotlib.pyplot as plt
from brian2 import *
from keras.datasets import mnist
from tqdm import tqdm
from sklearn import metrics
import numpy as np
import pandas as pd
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from spikingjelly import visualizing
import utils
import networkx as nx


n_input = 28 * 28  # input layer
n_e = 200  # e - excitatory
n_i = 2  # i - inhibitory

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
    w = clip(w + lr*Apost, 0, 1)'''
post = '''Apost += dApost
    w = clip(w + lr*Apre, 0, 1)'''


def generate_small_world(n_nodes, k, p):
    return nx.watts_strogatz_graph(n_nodes, k, p)


class Model:
    def __init__(self, debug=False):
        app = {}

        # инициализация малого мира
        G = generate_small_world(n_e, 10, 0.5)

        # Входные изображения кодируются как скорость Пуассоновских генераторов
        app['PG'] = PoissonGroup(n_input, rates=np.zeros(n_input) * Hz, name='PG')

        # Группа возбуждающих нейронов
        neuron_e = '''
            dv/dt = (ge*(0*mV-v) + gi*(-100*mV-v) + (v_rest_e-v)) / (100*ms) : volt
            dge/dt = -ge / (5*ms) : 1
            dgi/dt = -gi / (10*ms) : 1
            '''
        app['EG'] = NeuronGroup(n_e, neuron_e, threshold='v>v_thresh_e', refractory=5 * ms, reset='v=v_reset_e',
                                method='euler', name='EG')
        app['EG'].v = v_rest_e - 20. * mV

        if (debug):
            app['ESP'] = SpikeMonitor(app['EG'], name='ESP')
            app['ESM'] = StateMonitor(app['EG'], ['v'], record=True, name='ESM')
            app['ERM'] = PopulationRateMonitor(app['EG'], name='ERM')

        # Группа ингибирующих нейронов
        neuron_i = '''
            dv/dt = (ge*(0*mV-v) + (v_rest_i-v)) / (10*ms) : volt
            dge/dt = -ge / (5*ms) : 1
            '''
        app['IG'] = NeuronGroup(n_i, neuron_i, threshold='v>v_thresh_i', refractory=2 * ms, reset='v=v_reset_i',
                                method='euler', name='IG')
        app['IG'].v = v_rest_i - 20. * mV

        if (debug):
            app['ISP'] = SpikeMonitor(app['IG'], name='ISP')
            app['ISM'] = StateMonitor(app['IG'], ['v'], record=True, name='ISM')
            app['IRM'] = PopulationRateMonitor(app['IG'], name='IRM')

        # poisson generators one-to-all excitatory neurons with plastic connections
        app['S1'] = Synapses(app['PG'], app['EG'], stdp, on_pre=pre, on_post=post, method='euler', name='S1')
        app['S1'].connect()
        app['S1'].w = 'rand() * 0.3'  # random weights initialisation
        app['S1'].lr = 1  # enable stdp

        # здесь надо добавить другое правило обучения, а не STDP
        app['S_small_world'] = Synapses(app['EG'], app['EG'], stdp, on_pre=pre, on_post=post, method='euler', name='S_small_world')

        for (u, v) in G.edges():
            # print(f'{u} --- {v}')
            app['S_small_world'].connect(j=u, i=v)
            app['S_small_world'].w = 'rand() * 0.3'  # здесь нужно сделать так, чтобы генерировались
            # рандомные веса на порядок меньше, чем основные

        if (debug):
            # some synapses
            app['S1M'] = StateMonitor(app['S1'], ['w', 'Apre', 'Apost'], record=app['S1'][380, :4], name='S1M')

            # excitation neurons one-to-one inhibitory neurons
        app['S2'] = Synapses(app['EG'], app['IG'], 'w : 1', on_pre='ge += w', name='S2')
        # app['S2'].connect(j=0, i=range(100))
        # app['S2'].connect(j=1, i=range(100, 200))
        # app['S2'].delay = 'rand()*10*ms'
        # app['S2'].w = 'rand() * 0.3'  # random weights initialisation
        # app['S2'].lr = 1
        app['S2'].connect(j='i')
        app['S2'].delay = 'rand()*10*ms'
        app['S2'].w = 3
        # very strong fixed weights to ensure corresponding inhibitory neuron will always fire

        # inhibitory neurons one-to-all-except-one excitatory neurons
        app['S3'] = Synapses(app['IG'], app['EG'], 'w : 1', on_pre='gi += w', name='S3')
        app['S3'].connect(condition='i!=j')
        app['S3'].delay = 'rand()*5*ms'
        app['S3'].w = .03
        # weights are selected in such a way as to maintain a balance between excitation and ibhibition

        self.net = Network(app.values())
        self.net.run(0 * second)

    def __getitem__(self, key):
        return self.net[key]

    def train(self, X, X_test, num_of_examples=None, epoch=1):
        self.net['S1'].lr = 1
        self.net['S2'].lr = 1
        self.net['S_small_world'].lr = 1
        # stdp on

        features_train = [[] for _ in range(len(X))]
        start, stop = 0, 5000
        for ep in tqdm(range(epoch)):
            # data = X[num_of_examples * ep:num_of_examples * (ep + 1)]
            for idx in range(len(X)):
                # active mode
                #inp = SpikeMonitor(self.net['PG'], name='PM')
                #mon = SpikeMonitor(self.net['EG'], name='RM')
                #mon2 = SpikeMonitor(self.net['IG'], name='IM')
                #wei = StateMonitor(self.net['S1'], ['w', 'Apre', 'Apost'], record=self.net['S1'][380, :5], name='S1MM')

                #self.net.add(inp)
                #self.net.add(mon)
                #self.net.add(mon2)
                #self.net.add(wei)
                
                self.net['PG'].rates = X[idx].ravel() * Hz
                self.net.run(0.35 * second)

                #features_train[idx].append(np.array(mon.count, dtype=int8))
                # passive mode
                self.net['PG'].rates = np.zeros(n_input) * Hz
                self.net.run(0.15 * second)

                #if idx % num_of_examples == 0 and (ep > 25 or ep < 3): # сохранять только каждую num_of_examples итерацию
                    #plot_w(wei, f'weight/weight_{ep}_{idx}.png')
                    #plot_spikes(mon, mon2, f'spikes/spikes_{ep}_{idx}.png')
                    #plot_spikes(inp, mon, f'spikes/input_{ep}_{idx}.png')

                #self.net.remove(self.net['PM'])
                #self.net.remove(self.net['RM'])
                #self.net.remove(self.net['IM'])
                #self.net.remove(self.net['S1MM'])

                '''if idx % num_of_examples == 0:
                    visualizing.plot_2d_heatmap(array=(np.asarray(self.net['ESM'].v / b2.mV).T)[start:stop, :], title='Membrane Potentials',
                                                xlabel='Simulating Step',
                                                ylabel='Neuron Index', int_x_ticks=True, x_max=5000, dpi=200)

                    plt.savefig(f'spikes/heat_{ep}_{idx}.png')
                    plt.close()
                    start += 5000
                    stop += 5000'''

        self.net['S1'].lr = 0
        self.net['S2'].lr = 0
        self.net['S_small_world'].lr = 0

        features_test = []
        for idx in tqdm(range(len(X_test))):
            mon_test = SpikeMonitor(self.net['EG'], name='TestRM')
            mon2_test = SpikeMonitor(self.net['IG'], name='TestIM')

            self.net.add(mon_test)
            self.net.add(mon2_test)

            # active mode
            self.net['PG'].rates = X[idx].ravel() * Hz
            self.net.run(0.35 * second)

            # passive mode
            self.net['PG'].rates = np.zeros(n_input) * Hz
            self.net.run(0.15 * second)

            plot_spikes(mon_test, mon2_test, f'spikes/test_{idx}.png')
            # spikes per neuron foreach image
            features_test.append(np.array(mon2_test.count))

            self.net.remove(self.net['TestRM'])
            self.net.remove(self.net['TestIM'])

        return features_train, features_test

    def evaluate(self, X):
        self.net['S1'].lr = 0  # stdp off

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
            self.net['PG'].rates = np.zeros(n_input) * Hz
            self.net.run(0.15 * second)

            self.net.remove(self.net['RM'])

        return features


def plot_w(S1M, path):
    plt.rcParams["figure.figsize"] = (20, 10)
    subplot(311)
    plot(S1M.t / ms, S1M.w.T / gmax)
    ylabel('w / wmax')
    subplot(312)
    plot(S1M.t / ms, S1M.Apre.T)
    ylabel('apre')
    subplot(313)
    plot(S1M.t / ms, S1M.Apost.T)
    ylabel('apost')
    tight_layout()
    savefig(path)
    close();


def plot_v(ESM, ISM, neuron=13):
    plt.rcParams["figure.figsize"] = (20, 6)
    cnt = -50000  # tail
    plot(ESM.t[cnt:] / ms, ESM.v[neuron][cnt:] / mV, label='exc', color='r')
    plot(ISM.t[cnt:] / ms, ISM.v[neuron][cnt:] / mV, label='inh', color='b')
    plt.axhline(y=v_thresh_e / mV, color='pink', label='v_thresh_e')
    plt.axhline(y=v_thresh_i / mV, color='silver', label='v_thresh_i')
    legend()
    ylabel('v')
    show();


def plot_rates(ERM, IRM):
    plt.rcParams["figure.figsize"] = (20, 6)
    plot(ERM.t / ms, ERM.smooth_rate(window='flat', width=0.1 * ms) * Hz, color='r')
    plot(IRM.t / ms, IRM.smooth_rate(window='flat', width=0.1 * ms) * Hz, color='b')
    ylabel('Rate')
    show();


def plot_spikes(ESP, ISP, path):
    plt.rcParams["figure.figsize"] = (20, 6)
    plot(ESP.t / ms, ESP.i, '.r', label='exc')
    plot(ISP.t / ms, ISP.i, '.b', label='inh')
    ylabel('Neuron index')
    legend()
    savefig(path)
    close();


def test0(train_items=30):
    '''
    STDP visualisation
    '''
    seed(0)

    model = Model(debug=True)
    print(X_train[:train_items])
    model.train(X_train[:train_items], epoch=1)

    plot_w(model['S1M'])
    plot_v(model['ESM'], model['ISM'])
    plot_rates(model['ERM'], model['IRM'])
    plot_spikes(model['ESP'], model['ISP'])

def create_dataset(X_train, y_train, labels, num_of_ex):

    selected_matrices = []
    selected_labels = []
    selected_count = {label: 0 for label in set(y_train)}

    for matrix, label in zip(X_train, y_train):
        if selected_count[label] < num_of_ex and label in labels:
            selected_matrices.append(matrix)
            selected_labels.append(label)
            selected_count[label] += 1

    return np.array(selected_matrices), np.array(selected_labels)


def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # simplified classification (0 and 1)
    X_train = X_train[(y_train == 1) | (y_train == 0)]
    y_train = y_train[(y_train == 1) | (y_train == 0)]
    X_test = X_test[(y_test == 1) | (y_test == 0)]
    y_test = y_test[(y_test == 1) | (y_test == 0)]

    # pixel intensity to Hz (255 becoms ~63Hz)
    X_train = X_train / 4
    X_test = X_test / 4

    X_train.shape, X_test.shape

    # pixel intensity to Hz (255 becoms ~63Hz)
    X_train = X_train / 4
    X_test = X_test / 4

    selected_matrices = []
    selected_labels = []
    selected_count = {label: 0 for label in set(y_train)}

    for matrix, label in zip(X_train, y_train):
        if selected_count[label] < 10 and label in [7, 4]:
            selected_matrices.append(matrix)
            selected_labels.append(label)
            selected_count[label] += 1

    # Преобразуем списки в numpy массивы
    X_train, y_train = create_dataset(X_train, y_train, [0, 1], 20)
    X_test, y_test = create_dataset(X_test, y_test, [0, 1], 3)

    model = Model(True)
    assign_items = len(X_test)  # 60k

    seed(0)

    f_train, f_test = model.train(X_train, X_test, 20, epoch=20)

    for i, v in zip(f_test, y_test):
        print(i, v)


    clf = RandomForestClassifier(max_depth=4, random_state=0)
    clf.fit(f_train, y_train)

    print(clf.score(f_test, y_test))

    y_pred = clf.predict(f_test)
    conf_m = confusion_matrix(y_pred, y_test)
    print(conf_m)

    model.net.store('train', 'train.b2')

    # --- Датасет с данными о модели ---
    # time - время
    # exc_rate, inh_rate - частота спайкования возбуждающих и ингибирующих нейронов

    
    data_dict = {'time': model['ERM'].t / b2.ms,
                 'exc_rate': model['ERM'].smooth_rate(window='flat', width=0.1 * b2.ms) / b2.Hz,
                 'inh_rate': model['IRM'].smooth_rate(window='flat', width=0.1 * b2.ms) / b2.Hz}

    # (n_exc = n_inh)
    # n_exc_{i}, n_inh_{i} - мембранный потенциал возбуждающих и ингибирующих нейронов

    for i in range(n_e):
        data_dict[f'n_exc_{i}'] = model['ESM'].v[i] / mV
        data_dict[f'n_inh_{i}'] = model['ISM'].v[i] / mV

    labels = []
    for label in y_test:
        labels.append([label] * 5000)
    data_dict['labels'] = list(itertools.chain(*labels))

    dataframe = pd.DataFrame(data=data_dict)
    dataframe.to_csv('data.csv', index=False)

    print(model['ESM'].v / b2.mV)

    visualizing.plot_2d_heatmap(array=np.asarray(model['ESM'].v / b2.mV).T, title='Membrane Potentials', xlabel='Simulating Step',
                                ylabel='Neuron Index', int_x_ticks=True, x_max=5000, dpi=200)

    # visualizing.plot_2d_bar_in_3d(np.asarray(model['ESM'].v[0:6] / b2.mV).T, title='voltage of neurons', xlabel='neuron index',
    #                               ylabel='simulating step', zlabel='voltage', int_x_ticks=True, int_y_ticks=True,
    #                               int_z_ticks=True, dpi=200)

    plt.savefig('image.png')
    plt.show()
    plt.close()

    # utils.plot_rates(model['ERM'], model['IRM'])'''

    # model.net.restore()


if __name__ == '__main__':
    main()