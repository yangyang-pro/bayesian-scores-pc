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
from utils import eval_cnet_log_likes, eval_cnet_size, eval_cnet_bdeu_score

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
    parser = argparse.ArgumentParser(description='CNetBD Experiments')
    parser.add_argument('--ess', type=float, default=0.1, help='Equivalent sample size')
    parser.add_argument('--n-search-ors', type=int, default=10, help='Number of OR candidates')
    parser.add_argument('--learn-clt', choices=['MI', 'BD'], default='MI', help='Method of learning CLTs')
    parser.add_argument('--output', type=str, default='result/bcnet', help='Output store path')
    args = parser.parse_args()

    ess = args.ess
    n_search_ors = args.n_search_ors
    method_clt = args.learn_clt
    output_path = args.output

    time_dir = os.path.join(output_path, datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    os.makedirs(time_dir, exist_ok=True)
    ll_dir = os.path.join(time_dir, 'lls')
    os.makedirs(ll_dir, exist_ok=True)
    model_dir = os.path.join(time_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    csv_file = open(os.path.join(time_dir, 'bcnet.csv'), 'w')
    fieldnames = ['Dataset',
                  'marginal_ll', 'train_ll', 'test_ll',
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
        print('test size', test.shape)

        cnet = CNet()
        start = time.time()
        cnet.bounded_search(data=train, ess=ess, n_search_ors=n_search_ors, method_clt=method_clt)
        cnet_time = time.time() - start

        bdeu_score = eval_cnet_bdeu_score(root=cnet.root, data=train)
        train_lls = eval_cnet_log_likes(root=cnet.root, data=train)
        test_lls = eval_cnet_log_likes(root=cnet.root, data=test)
        mean_train_ll = np.mean(train_lls)
        mean_test_ll = np.mean(test_lls)
        mean_marginal_ll = bdeu_score / len(train)

        print(dataset)
        print('{:<30}{:.3f}'.format('Learning time:', cnet_time))

        print('{:<30}{:.3f}'.format('Mean marginal LL:', mean_marginal_ll))
        print('{:<30}{:.3f}'.format('Mean training LL:', mean_train_ll))
        print('{:<30}{:.3f}'.format('Mean test LL:', mean_test_ll))

        cnet_statistics = eval_cnet_size(root=cnet.root)

        print()
        for k, v in cnet_statistics.items():
            print('{:<30}{:<}'.format(k + ':', v))

        with open(os.path.join(model_dir, dataset + '.bcnet'), 'wb') as cnet_file:
            pickle.dump(cnet, cnet_file)

        np.savetxt(os.path.join(ll_dir, dataset + '.train.lls'), train_lls)
        np.savetxt(os.path.join(ll_dir, dataset + '.test.lls'), test_lls)

        info = {'Dataset': dataset,
                'time': '{:.3f}'.format(cnet_time),
                'marginal_ll': '{:.3f}'.format(mean_marginal_ll),
                'train_ll': '{:.3f}'.format(mean_train_ll),
                'test_ll': '{:.3f}'.format(mean_test_ll)}
        info.update(cnet_statistics)
        writer.writerow(info)
        csv_file.flush()
    csv_file.close()

