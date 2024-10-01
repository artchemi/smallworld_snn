import numpy as np
from tqdm import tqdm
from keras.datasets import mnist
from models.snn_200_paper import Model as snn_paper
from models.snn_200_sw import Model as snn_sw
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from scipy.interpolate import interp1d


(X_train, y_train), (X_test, y_test) = mnist.load_data()

path_main_tmp = 'data_15_09_2024_180423/'
path_main = path_main_tmp + 'epoch_0/'

def plot_patterns(patterns_dict: dict, save_folder: str) -> None:
    plt.rcParams['figure.figsize'] = [20, 8]
    for i in range(0, 10):
        plt.subplot(2, 5, i+1)
        plt.imshow(patterns_dict[f'label_{i}'])
        plt.title(f'Label {i}')
    
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes((0.85, 0.1, 0.075, 0.8))
    plt.colorbar(cax=cax)

    plt.savefig(save_folder + 'patterns_vis.png')
    
    plt.show()


def ml_classifier(path: str) -> dict:
    metrics = {}
    
    for epoch in tqdm(range(2)):
        rf_model = RandomForestClassifier(max_depth=5, random_state=42)

        X_train = []
        y_train = []

        X_test = []
        y_test = []

        for i in range(0, 10):
            with open(path + f'/epoch_{epoch}/' + f'rates_{i}_label.npy', 'rb') as file:
                data = np.load(file)

            with open(path + f'/epoch_{epoch}/' + f'rates_test/rates_{i}_label.npy', 'rb') as file:
                data_test = np.load(file)

            for rate in data:
                X_train.append(np.reshape(rate, 100))
                y_train.append(i)

            for rate_test in data_test:
                X_test.append(np.reshape(rate_test, 100))
                y_test.append(i)

        # X_train, y_train = shuffle(X_train, y_train, random_state=42)

        rf_model.fit(X_train, y_train)

        y_train_pred = rf_model.predict(X_train)
        y_test_pred = rf_model.predict(X_test)

        if epoch == 1:
            metrics[f'rf_train_{epoch}'] = rf_model.score(X_train, y_train)
            metrics[f'rf_test_{epoch}'] = rf_model.score(X_test, y_test)

            metrics[f'rf_train_precision_{epoch}'] = precision_score(y_train_pred, y_train, average='macro')
            metrics[f'rf_test_precision_{epoch}'] = precision_score(y_test_pred, y_test, average='macro')

            metrics[f'rf_train_recall_{epoch}'] = recall_score(y_train_pred, y_train, average='macro')
            metrics[f'rf_test_recall_{epoch}'] = recall_score(y_test_pred, y_test, average='macro')
    
    # cm = confusion_matrix(y_train_pred, y_train)
    # print(cm)

    return metrics


def main():
    patterns = {}
    
    for n in range(0, 10):
        path = path_main + f'rates_{n}_label.npy'

        with open(path, 'rb') as file:
            data = np.load(file)

        # Вычисление усредненного паттерна
        for i in range(data.shape[0]):
            if i == 0:
                m = data[i]
            else:
                m = m + data[i]

        pattern_mean = np.round(m / data.shape[0], 2)

        patterns[f'label_{n}'] = pattern_mean

    
    for i in range(0, 10):
        path = path_main + f'rates_{i}_label.npy'
        true_label = [i]
        
        eval_data = []
        with open(path, 'rb') as file:
            data = np.load(file)

        for j in range(data.shape[0]):
            diff_lst_temp = []
            for n in range(0, 10):
                diff = np.sum(np.round(patterns[f'label_{n}'] - data[j], 1))
                diff_lst_temp.append(np.absolute(diff))

            pred_label = np.argmin(diff_lst_temp)
            eval_data.append(true_label + diff_lst_temp + [pred_label])

    df = pd.DataFrame(data=np.asarray(eval_data), columns=['true_label', 'diff_0', 'diff_1', 'diff_2', 
                                                           'diff_3', 'diff_4', 'diff_5', 'diff_6', 
                                                           'diff_7', 'diff_8', 'diff_9', 'pred_label'])
    
    
    print(ml_classifier(path_main_tmp))

    #plot_patterns(patterns_dict=patterns, save_folder=path_main)

    # print(ml_classifier(path_main_tmp))

    # acc_lst = []
    # acc_test_lst = []

    # precision_train_lst = []
    # precision_test_lst = []

    # recall_train_lst = []
    # recall_test_lst = []

    # for experiment in ['data_06_09_2024_022618', 'data_06_09_2024_022750', 
    #                    'data_06_09_2024_022802', 'data_06_09_2024_022813',
    #                    'data_29_08_2024_222249']:
        
    #     path_main_tmp = experiment + '/'
    #     acc_dict = ml_classifier(path_main_tmp)

    #     acc_train_value, acc_test_value = acc_dict['rf_train_1'], acc_dict['rf_test_1']
    #     precision_train_value, precision_test_value = acc_dict['rf_train_precision_1'], acc_dict['rf_test_precision_1']
    #     recall_train_value, recall_test_value = acc_dict['rf_train_recall_1'], acc_dict['rf_test_recall_1'] 

    #     acc_lst.append(acc_train_value)
    #     acc_test_lst.append(acc_test_value)

    #     precision_train_lst.append(precision_train_value)
    #     precision_test_lst.append(precision_test_value)

    #     recall_train_lst.append(recall_train_value)
    #     recall_test_lst.append(recall_test_value)

    # lr_lst = [0.2, 0.4, 0.6, 0.8, 1.0]

    # f_interp_train = interp1d(lr_lst, acc_lst, kind='cubic')
    # f_interp_test = interp1d(lr_lst, acc_test_lst, kind='cubic')

    # f_interp_precision_train = interp1d(lr_lst, precision_train_lst, kind='cubic')
    # f_interp_precision_test = interp1d(lr_lst, precision_test_lst, kind='cubic')

    # f_interp_recall_train = interp1d(lr_lst, recall_train_lst, kind='cubic')
    # f_interp_recall_test = interp1d(lr_lst, recall_test_lst, kind='cubic')

    # lr_inter = np.linspace(0.2, 1, 100)

    # # --- accuracy ---
    # plt.scatter(lr_lst, acc_lst, label='Accuracy, train', color='black')
    # plt.plot(lr_inter, f_interp_train(lr_inter), color='black')

    # plt.scatter(lr_lst, acc_test_lst, label='Accuracy, test', color='red')
    # plt.plot(lr_inter, f_interp_test(lr_inter), color='red')

    # # --- precision ---
    # plt.scatter(lr_lst, precision_train_lst, label='Precision, train', color='black', marker='*')
    # plt.plot(lr_inter, f_interp_precision_train(lr_inter), linestyle='--', color='black')

    # plt.scatter(lr_lst, precision_test_lst, label='Precision, test', color='red', marker='*')
    # plt.plot(lr_inter, f_interp_precision_test(lr_inter), linestyle='--', color='red')

    # # --- recall ---
    # plt.scatter(lr_lst, recall_train_lst, label='Recall, train', color='black', marker='P')
    # plt.plot(lr_inter, f_interp_recall_train(lr_inter), linestyle='-.', color='black')

    # plt.scatter(lr_lst, recall_test_lst, label='Recall, test', color='red', marker='P')
    # plt.plot(lr_inter, f_interp_recall_test(lr_inter), linestyle='-.', color='red')

    # for i in range(0, 5):
    #     print(f'--- learning_rate = {lr_lst[i]} ---')
    #     print(f'Accuracy: train = {acc_lst[i]} | test = {acc_test_lst[i]}')
    #     print(f'Precision: train = {precision_train_lst[i]} | test = {precision_test_lst[i]}')
    #     print(f'Recall: train = {recall_test_lst[i]} | test = {recall_train_lst[i]}')


    # plt.xlabel('Learning Rate')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()

        



if __name__ == '__main__':
    main()