from os import path

import torch
from numpy import genfromtxt
import scipy.stats as stats
import numpy as np
from torch.utils.data import Dataset


class UnivariateDataset(Dataset):
    def __init__(self, dataset_path, batch_size=1, is_train=1, noise=0.0):
        raw_arr = genfromtxt(dataset_path, delimiter='\t')
        self.data = raw_arr[:, 1:]
        self.check_z_norm()
        self.batch_size = batch_size
        self.size = int(np.ceil(len(self.data)/self.batch_size))
        self.labels = raw_arr[:, 0] - 1
        self.check_labels()
        self.is_train = is_train
        self.noise = noise
        self.number_of_instance = len(self.data)

    def check_z_norm(self):
        if np.abs(np.round(np.mean(self.data[0]),3)) > 0:
            self.data = stats.zscore(self.data, axis=1, ddof=1)
            # for i in range(len(self.data)):
            #     self.data[i] = stats.zscore(self.data[i])

    def check_labels(self):
        labels_name = np.unique(self.labels)
        for i in range(len(self.labels)):
            for j in range(len(labels_name)):
                if self.labels[i] == labels_name[j]:
                    self.labels[i] = j
                    break

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx == self.size - 1:
            return_data = self.data[idx * self.batch_size:]
            return_label = self.labels[idx * self.batch_size:]
        else:
            return_data = self.data[idx*self.batch_size:(idx+1)*self.batch_size]
            return_label = self.labels[idx*self.batch_size:(idx+1)*self.batch_size]
        if self.is_train:
            random_noise = np.random.normal(0,self.noise,size=(return_data.shape[0], return_data.shape[-1]))
            return_data += random_noise

        return_data = torch.FloatTensor(return_data)
        return_data = return_data.view(return_data.size(0),1,return_data.size(-1))
        return_label = torch.LongTensor(return_label)
        return return_data, return_label

    def change_device(self, device):
        self.data = self.data.to(device)
        self.labels = self.labels.to(device)

    def update_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.size = int(np.ceil(len(self.data)/self.batch_size))



def load_dataset(dataset_name, dataset_folder):
    dataset_path = path.join(dataset_folder, dataset_name)
    train_file_path = path.join(dataset_path, '{}_TRAIN'.format(dataset_name))
    test_file_path = path.join(dataset_path, '{}_TEST'.format(dataset_name))

    # training data
    train_raw_arr = genfromtxt(train_file_path, delimiter=',')
    train_data = train_raw_arr[:, 1:]
    train_labels = train_raw_arr[:, 0] - 1
    # one was subtracted to change the labels to 0 and 1 instead of 1 and 2

    # test_data
    test_raw_arr = genfromtxt(test_file_path, delimiter=',')
    test_data = test_raw_arr[:, 1:]
    test_labels = test_raw_arr[:, 0]
    while min(test_labels) < 0:
        test_labels += 1
        train_labels += 1

    return train_data, train_labels, test_data, test_labels


def load_dataset_zscore(dataset_name, dataset_folder):
    dataset_path = path.join(dataset_folder, dataset_name)
    train_file_path = path.join(dataset_path, '{}_TRAIN'.format(dataset_name))
    test_file_path = path.join(dataset_path, '{}_TEST'.format(dataset_name))

    # training data
    train_raw_arr = genfromtxt(train_file_path, delimiter=',')
    train_data = train_raw_arr[:, 1:]
    train_labels = train_raw_arr[:, 0] - 1
    # one was subtracted to change the labels to 0 and 1 instead of 1 and 2

    # test_data
    test_raw_arr = genfromtxt(test_file_path, delimiter=',')
    test_data = test_raw_arr[:, 1:]
    test_labels = test_raw_arr[:, 0] - 1

    z_train_data = np.asarray([stats.zscore(data) for data in train_data])
    z_test_data = np.asarray([stats.zscore(data) for data in test_data])

    return z_train_data, train_labels, z_test_data, test_labels


def load_dataset_varylen(dataset_name, dataset_folder):
    dataset_path = path.join(dataset_folder, dataset_name)
    train_file_path = path.join(dataset_path, '{}_TRAIN.tsv'.format(dataset_name))
    test_file_path = path.join(dataset_path, '{}_TEST.tsv'.format(dataset_name))

    # training data
    train_raw_arr = genfromtxt(train_file_path, delimiter='\t')
    train_data = train_raw_arr[:, 1:]
    train_labels = train_raw_arr[:, 0] - 1
    # one was subtracted to change the labels to 0 and 1 instead of 1 and 2

    # test_data
    test_raw_arr = genfromtxt(test_file_path, delimiter='\t')
    test_data = test_raw_arr[:, 1:]
    test_labels = test_raw_arr[:, 0] - 1

    new_train_data = [data[np.isnan(data) == False] for data in train_data]
    new_test_data = [data[np.isnan(data) == False] for data in test_data]

    return new_train_data, train_labels, new_test_data, test_labels


def sort_data_by_error_list(data, error_list):
    error_data = []
    correct_data = []
    for i in range(len(error_list)):
        if error_list[i]:
            correct_data.append(data[i])
        else:
            error_data.append(data[i])
    new_data = error_data + correct_data
    if len(error_list) < len(data):
        for i in range(len(error_list), len(data)):
            new_data.append(data[i])

    return new_data

