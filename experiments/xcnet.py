import os
import sys

sys.path.append('..')

import csv
import time
import datetime
import pickle
import argparse
import itertools

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
    sys.setrecursionlimit(10000)
    parser = argparse.ArgumentParser(description='XCNet Experiments')
    parser.add_argument('--alpha', type=float, nargs='+', default=[0.5], help='Laplace smoothing factor')
    parser.add_argument('--min-n-samples', type=int, nargs='+', default=[500],
                        help='Minimum number of instances to split')
    parser.add_argument('--min-n-features', type=int, nargs='+', default=[3],
                        help='Minimum number of features to split')
    parser.add_argument('--n-runs', type=int, default=10, help='Number of runs to train XCNet')
    parser.add_argument('--store-models', action='store_true')
    parser.add_argument('--output', type=str, default='result/xcnet/', help='Output store path')
    args = parser.parse_args()

    min_n_sample_cands = args.min_n_samples
    min_n_feature_cands = args.min_n_features
    n_runs = args.n_runs
    alphas = args.alpha
    is_store_models = args.store_models
    output_path = args.output

    time_dir = os.path.join(output_path, datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    os.makedirs(time_dir, exist_ok=True)
    ll_dir = os.path.join(time_dir, 'lls')
    os.makedirs(ll_dir, exist_ok=True)

    summary_csv_path = os.path.join(time_dir, 'result.csv')
    fieldnames = ['Dataset', 'time', 'alpha', 'min_n_samples', 'min_n_features',
                  'train_ll', 'valid_ll', 'test_ll']
    with open(summary_csv_path, 'w') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        file.flush()

    for dataset in BINARY_DATASETS:
        dataset_dir = os.path.join(DATASETS_PATH, dataset)
        train_path = os.path.join(dataset_dir, dataset + '.train.data')
        valid_path = os.path.join(dataset_dir, dataset + '.valid.data')
        test_path = os.path.join(dataset_dir, dataset + '.test.data')
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

        result_dataset_dir = os.path.join(time_dir, dataset)
        os.makedirs(result_dataset_dir, exist_ok=True)

        csv_path = os.path.join(result_dataset_dir, dataset + '.csv')
        sub_fieldnames = ['time', 'alpha', 'min_n_samples', 'min_n_features',
                          'train_ll', 'valid_ll', 'test_ll']
        with open(csv_path, 'w') as file:
            writer = csv.DictWriter(file, fieldnames=sub_fieldnames)
            writer.writeheader()
            file.flush()

        best_mean_train_lls = None
        best_mean_valid_lls = None
        best_mean_test_lls = None
        best_mean_train_ll = None
        best_mean_valid_ll = -np.inf
        best_mean_test_ll = None
        best_alpha = None
        best_min_n_samples = None
        best_min_n_features = None
        best_total_time = None
        for alpha, min_n_samples, min_n_features in itertools.product(alphas, min_n_sample_cands, min_n_feature_cands):
            all_train_lls = []
            all_valid_lls = []
            all_test_lls = []
            cnet_times = []
            hypers_dir = os.path.join(result_dataset_dir, str(alpha) + '_' + str(min_n_samples)
                                      + '_' + str(min_n_features))
            os.makedirs(hypers_dir, exist_ok=True)
            runs_csv_path = os.path.join(hypers_dir, 'runs.csv')
            runs_fieldnames = ['time', 'alpha', 'min_n_samples', 'min_n_features',
                               'train_ll', 'valid_ll', 'test_ll',
                               'cnet_depth', 'n_or', 'n_clts', 'n_params']
            with open(runs_csv_path, 'w') as file:
                writer = csv.DictWriter(file, fieldnames=runs_fieldnames)
                writer.writeheader()
                file.flush()
            for i in range(n_runs):
                cnet = CNet()
                start = time.time()
                cnet.learn_xcnet(data=train, alpha=alpha, min_n_samples=min_n_samples, min_n_features=min_n_features)
                cnet_time = time.time() - start
                cnet_times.append(cnet_time)
                train_lls = eval_cnet_log_likes(root=cnet.root, data=train)
                valid_lls = eval_cnet_log_likes(root=cnet.root, data=valid)
                test_lls = eval_cnet_log_likes(root=cnet.root, data=test)
                all_train_lls.append(train_lls)
                all_valid_lls.append(valid_lls)
                all_test_lls.append(test_lls)
                if is_store_models:
                    cnet_statistics = eval_cnet_size(root=cnet.root)
                    info = {'alpha': alpha,
                            'min_n_samples': min_n_samples,
                            'min_n_features': min_n_features,
                            'time': '{:.3f}'.format(cnet_time),
                            'train_ll': '{:.3f}'.format(np.mean(train_lls)),
                            'valid_ll': '{:.3f}'.format(np.mean(valid_lls)),
                            'test_ll': '{:.3f}'.format(np.mean(test_lls))}
                    info.update(cnet_statistics)
                    with open(runs_csv_path, 'a') as file:
                        writer = csv.DictWriter(file, fieldnames=runs_fieldnames)
                        writer.writerow(info)
                        file.flush()
                    with open(os.path.join(hypers_dir, dataset + '_No.' + str(i) + '.xcnet'), 'wb') as file:
                        pickle.dump(cnet, file)

            total_cnet_time = np.sum(cnet_times)

            mean_train_lls = np.mean(all_train_lls, axis=0)
            mean_valid_lls = np.mean(all_valid_lls, axis=0)
            mean_test_lls = np.mean(all_test_lls, axis=0)

            mean_train_ll = np.mean(mean_train_lls)
            mean_valid_ll = np.mean(mean_valid_lls)
            mean_test_ll = np.mean(mean_test_lls)

            if mean_valid_ll > best_mean_valid_ll:
                best_mean_train_ll = mean_train_ll
                best_mean_valid_ll = mean_valid_ll
                best_mean_test_ll = mean_test_ll
                best_alpha = alpha
                best_min_n_samples = min_n_samples
                best_min_n_features = min_n_features
                best_mean_train_lls = mean_train_lls
                best_mean_valid_lls = mean_valid_lls
                best_mean_test_lls = mean_test_lls
                best_total_time = total_cnet_time

            print('-' * 60)
            print(dataset)
            print('{:<30}{:.3f}'.format('Learning time:', total_cnet_time))

            print('{:<30}{:.3f}'.format('Mean training LL:', mean_train_ll))
            print('{:<30}{:.3f}'.format('Mean validation LL:', mean_valid_ll))
            print('{:<30}{:.3f}'.format('Mean test LL:', mean_test_ll))
            print('-' * 60)

            info = {'alpha': alpha,
                    'min_n_samples': min_n_samples,
                    'min_n_features': min_n_features,
                    'time': '{:.3f}'.format(total_cnet_time),
                    'train_ll': '{:.3f}'.format(mean_train_ll),
                    'valid_ll': '{:.3f}'.format(mean_valid_ll),
                    'test_ll': '{:.3f}'.format(mean_test_ll)}
            with open(csv_path, 'a') as file:
                writer = csv.DictWriter(file, fieldnames=sub_fieldnames)
                writer.writerow(info)
                file.flush()
        info = {'Dataset': dataset,
                'time': '{:.3f}'.format(best_total_time),
                'alpha': best_alpha,
                'min_n_samples': best_min_n_samples,
                'min_n_features': best_min_n_features,
                'train_ll': '{:.3f}'.format(best_mean_train_ll),
                'valid_ll': '{:.3f}'.format(best_mean_valid_ll),
                'test_ll': '{:.3f}'.format(best_mean_test_ll)}
        with open(summary_csv_path, 'a') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow(info)
            file.flush()

        np.savetxt(os.path.join(ll_dir, dataset + '.train.lls'), best_mean_train_lls)
        np.savetxt(os.path.join(ll_dir, dataset + '.valid.lls'), best_mean_valid_lls)
        np.savetxt(os.path.join(ll_dir, dataset + '.test.lls'), best_mean_test_lls)
