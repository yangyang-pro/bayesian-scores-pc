import os
import sys
sys.path.append('..')

import csv
import time
import datetime
import pickle
import argparse

import numpy as np

from cnet import CNet
from utils import eval_cnet_log_likes, eval_cnet_size

BINARY_DATASETS = [
    'nltcs',
    'msnbc',
    'kdd',
    'plants',
    'baudio',
    'jester',
    'bnetflix',
    'accidents',
    'tretail',
    'pumsb_star',
    'dna',
    'kosarek',
    'msweb',
    'book',
    'tmovie',
    'cwebkb',
    'cr52',
    'c20ng',
    'bbc',
    'ad',
]

DATASETS_PATH = 'binary_datasets'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Original CNet Experiments')
    parser.add_argument('--min-n-samples', type=int, default=10, help='Minimum number of instances to split')
    parser.add_argument('--min-n-features', type=int, default=1, help='Minimum number of features to split')
    parser.add_argument('--alpha', type=float, default=1.0, help='Smoothing factor')
    parser.add_argument('--output', type=str, default='result/cnet_entropy', help='Output store path')
    args = parser.parse_args()

    min_n_samples = args.min_n_samples
    min_n_features = args.min_n_features
    alpha = args.alpha
    output_path = args.output

    time_dir = os.path.join(output_path, datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    os.makedirs(time_dir, exist_ok=True)
    ll_dir = os.path.join(time_dir, 'lls')
    os.makedirs(ll_dir, exist_ok=True)
    model_dir = os.path.join(time_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    csv_file = open(os.path.join(time_dir, 'cnet_entropy.csv'), 'w')
    fieldnames = ['Dataset',
                  'train_ll', 'valid_ll', 'test_ll',
                  'time', 'cnet_depth', 'n_or', 'n_clts', 'n_params']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for dataset in BINARY_DATASETS:
        dataset_path = os.path.join(DATASETS_PATH, dataset)
        train_path = os.path.join(dataset_path, dataset + '.train.data')
        valid_path = os.path.join(dataset_path, dataset + '.valid.data')
        test_path = os.path.join(dataset_path, dataset + '.test.data')
        with open(train_path, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            train = np.array(list(reader)).astype(np.float32)
        with open(valid_path, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            valid = np.array(list(reader)).astype(np.float32)
        with open(test_path, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            test = np.array(list(reader)).astype(np.float32)
        train = np.vstack((train, valid))
        n_samples, n_features = train.shape
        print(dataset)

        print('train size', train.shape)
        print('valid size', valid.shape)
        print('test size', test.shape)

        cnet = CNet()
        start = time.time()
        cnet.learn_cnet_entropy(data=train, min_n_samples=min_n_samples, min_n_features=min_n_features, alpha=alpha)
        cnet_time = time.time() - start

        train_lls = eval_cnet_log_likes(root=cnet.root, data=train)
        valid_lls = eval_cnet_log_likes(root=cnet.root, data=valid)
        test_lls = eval_cnet_log_likes(root=cnet.root, data=test)
        mean_train_ll = np.mean(train_lls)
        mean_valid_ll = np.mean(valid_lls)
        mean_test_ll = np.mean(test_lls)

        print(dataset)
        print('{:<30}{:.3f}'.format('Learning time:', cnet_time))

        print('{:<30}{:.3f}'.format('Mean training LL:', mean_train_ll))
        print('{:<30}{:.3f}'.format('Mean validation LL:', mean_valid_ll))
        print('{:<30}{:.3f}'.format('Mean test LL:', mean_test_ll))

        cnet_statistics = eval_cnet_size(root=cnet.root)

        print()
        for k, v in cnet_statistics.items():
            print('{:<30}{:<}'.format(k + ':', v))

        with open(os.path.join(model_dir, dataset + '.cnet_entropy'), 'wb') as cnet_file:
            pickle.dump(cnet, cnet_file)

        np.savetxt(os.path.join(ll_dir, dataset + '.test.lls'), test_lls)

        info = {'Dataset': dataset,
                'time': '{:.3f}'.format(cnet_time),
                'train_ll': '{:.3f}'.format(mean_train_ll),
                'valid_ll': '{:.3f}'.format(mean_valid_ll),
                'test_ll': '{:.3f}'.format(mean_test_ll)}
        info.update(cnet_statistics)
        writer.writerow(info)
        csv_file.flush()
    csv_file.close()

