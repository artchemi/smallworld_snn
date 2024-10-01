import numpy as np
from tqdm import tqdm
from keras.datasets import mnist
from models.snn_200_paper import Model as snn_paper
from models.snn_200_sw import Model as snn_sw
import argparse

import os
import multiprocessing


parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('--folder', default='data_29_08_2024_222940', type=str, 
                    help='Папка с весами')

parser.add_argument('--type', type=str, 
                    help='Тип модели')

parser.add_argument('--eval_type', type=str, 
                    help='Train/test оценка модели')

parser.add_argument('--train_size', default=60000, type=int,
                    help='Размер обучающей выборки')

parser.add_argument('--test_size', default=20000, type=int,
                    help='Размер тестовой выборки')

args = parser.parse_args()


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train[:args.train_size]
y_train = y_train[:args.train_size]

X_test = X_test[:args.test_size]
y_test = y_test[:args.test_size]

if args.type == 'paper':
    model = snn_paper(784, 100, args.folder, 1)
else:
    with open(args.folder + '/data.txt', 'r') as file:
        data = file.read()

    s_with_prob = None
    data = data.split('\n')

    for s in data:
        if 'Prob' in s:
            s_with_prob = s
            break

    probability = float(s_with_prob.split()[1])

    model = snn_sw(784, 100, args.folder, 0.2, probability)

def calculate_rates_0():
    print('Running calcs for epoch #0...')

    try:
        os.mkdir(f'{args.folder}/epoch_0/membrane_potential')
    except:
        None

    for i in tqdm(range(0, 10)):
        indexes_temp = np.where(y_train==i)
        X_train_temp = X_train[indexes_temp]

        features_lst_temp = []

        # count = 0

        for X in tqdm(X_train_temp):
            f_train, _ = model.evaluate(X, chk=args.folder + '/epoch_0/chk_0.b2')

            features_lst_temp.append(f_train)

            # with open(f'{args.folder}/epoch_0/membrane_potential/mem_pot_{i}_label_count_{count}.npy', 'wb') as file:
            #     np.save(file, np.array(mem_features_lst_temp))

            # count += 1

        with open(f'{args.folder}/epoch_0/rates_{i}_label.npy', 'wb') as file:
            np.save(file, np.array(features_lst_temp))

    
def calculate_rates_0_test():
    try:
        os.mkdir(args.folder + '/epoch_0/rates_test')
    except FileExistsError:
        None

    print('Running calcs for epoch #0...')
    for i in tqdm(range(0, 10)):
        indexes_temp = np.where(y_test==i)
        X_test_temp = X_test[indexes_temp]

        features_lst_temp = []

        for X in tqdm(X_test_temp):
            f_train, _ = model.evaluate(X, chk=args.folder + '/epoch_0/chk_0.b2')

            features_lst_temp.append(f_train)

        with open(f'{args.folder}/epoch_0/rates_test/rates_{i}_label.npy', 'wb') as file:
            np.save(file, np.array(features_lst_temp))



def calculate_rates_1():
    print('Running calcs for epoch #1...')

    try:
        os.mkdir(f'{args.folder}/epoch_1/membrane_potential')
    except:
        None

    for i in tqdm(range(0, 10)):
        indexes_temp = np.where(y_train==i)
        X_train_temp = X_train[indexes_temp]

        features_lst_temp = []

        for X in tqdm(X_train_temp):
            f_train, _ = model.evaluate(X, chk=args.folder + '/epoch_1/chk_1.b2')

            features_lst_temp.append(f_train)

        with open(f'{args.folder}/epoch_1/rates_{i}_label.npy', 'wb') as file:
            np.save(file, np.array(features_lst_temp))


def calculate_rates_1_test():
    try:
        os.mkdir(args.folder + '/epoch_1/rates_test')
    except FileExistsError:
        None

    print('Running calcs for epoch #1...')
    for i in tqdm(range(0, 10)):
        indexes_temp = np.where(y_test==i)
        X_test_temp = X_test[indexes_temp]

        features_lst_temp = []

        for X in tqdm(X_test_temp):
            f_train, _ = model.evaluate(X, chk=args.folder + '/epoch_1/chk_1.b2')

            features_lst_temp.append(f_train)

        with open(f'{args.folder}/epoch_1/rates_test/rates_{i}_label.npy', 'wb') as file:
            np.save(file, np.array(features_lst_temp))



def main():
    if args.eval_type == 'train':
        process1 = multiprocessing.Process(target=calculate_rates_0)
        process2 = multiprocessing.Process(target=calculate_rates_1)
    elif args.eval_type == 'test':
        process1 = multiprocessing.Process(target=calculate_rates_0_test)
        process2 = multiprocessing.Process(target=calculate_rates_1_test)
 
    process1.start()
    process2.start()
 
    process1.join()
    process2.join()



if __name__ == '__main__':
    main()

