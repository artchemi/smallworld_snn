from models import snn_200_paper
from models.snn_200_paper import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import argparse
import os
import time

from tqdm import tqdm

from brian2 import *

from analysis_rates import ml_classifier


parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')


parser.add_argument('-e', '--epochs', default=1, type=int, 
                    help='Количество эпох для обучения')

parser.add_argument('--hidden', default=100, type=int, 
                    help='Количество нейронов в скрытом слое')

parser.add_argument('--learning_rate', default=1, type=float, 
                    help='Скорость обучения')

parser.add_argument('--delta_tau', default=20, type=float, 
                    help='Delta TAU')

parser.add_argument('--train_size', default=60000, type=int,
                    help='Размер обучающей выборки')

parser.add_argument('--test_size', default=20000, type=int,
                    help='Размер тестовой выборки')

# python train.py --learning_rate 0.2 --delta_tau 10 --train_size 100 --test_size 100

args = parser.parse_args()


def main():
    parser.print_help()
    seed(42)

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train[:args.train_size]

    current_time = time.time()
    local_time = time.localtime(current_time)
    formatted_time = time.strftime('%d_%m_%Y_%H%M%S', local_time)

    dir_name = f'data_{formatted_time}'

    # set_device("cuda_standalone", build_on_run=True, compile=True, run=True, 
    #            debug=False)
    
    snn_200_paper.taupre = args.delta_tau * ms

    os.mkdir(dir_name)
    with open(f'{dir_name}/data.txt', 'a') as file:
        file.write(f'Work folder: {dir_name}\n')
        file.write('Model from paper\n')
        file.write(f'Train len: {len(X_train)}\n')
        file.write(f'Neurons in hidden layer: {args.hidden}\n')
        file.write(f'Learning rate: {args.learning_rate}\n')
        file.write(f'Delta TAU: {args.delta_tau}\n')
        file.close()

    os.mkdir(dir_name + '/weights')
    os.mkdir(dir_name + '/epoch_0')

    model = Model(784, args.hidden, dir_name, lr=args.learning_rate, save_flag=True)

    model.net.store('train', f'{dir_name}/epoch_0/chk_0.b2')

    os.mkdir(dir_name + '/epoch_1')

    with open(f'{dir_name}/data.txt', 'a') as file:
        file.write('Training...\n')

    model.train(X_train, args.epochs, weight_save=False)

    with open(f'{dir_name}/data.txt', 'a') as file:
        file.write('Training done!\n')
        file.write('Evaluating...\n')

    os.system(f'python eval_rates.py --folder {dir_name} --type "paper" --eval_type "train" --train_size {args.train_size} --test_size {args.test_size}')
    os.system(f'python eval_rates.py --folder {dir_name} --type "paper" --eval_type "test" --train_size {args.train_size} --test_size {args.test_size}')
    

    with open(f'{dir_name}/data.txt', 'a') as file:
        file.write('Evaluating done!\n')

    acc_dict = ml_classifier(dir_name)

    with open(f'{dir_name}/data.txt', 'a') as file:
        file.write(f'Train accuracy: {acc_dict["rf_train_1"]}\n')
        file.write(f'Test accuracy: {acc_dict["rf_test_1"]}\n')
        file.write(f'Train precision: {acc_dict["rf_train_precision_1"]}\n')
        file.write(f'Test precision: {acc_dict["rf_test_precision_1"]}\n')
        file.write(f'Train recall: {acc_dict["rf_train_recall_1"]}\n')
        file.write(f'Test recall: {acc_dict["rf_test_recall_1"]}\n')

if __name__ == '__main__':
    main()