import os
import sys
sys.path.append('..')

import csv
import time
import datetime
import pickle
import argparse

import numpy as np

from scipy.special import logsumexp

from cnet import MCNet
from utils import eval_cnet_log_likes

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
    'binarized_mnist',
]

DATASETS_PATH = 'binary_datasets'

if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    parser = argparse.ArgumentParser(description='EM CNetBD Experiments')
    parser.add_argument('dataset', choices=BINARY_DATASETS, help='Dataset')
    parser.add_argument('-k', type=int, nargs='+', default=[1, 2, 3, 5, 8, 10, 20], help='Number of EM components')
    parser.add_argument('--ess', type=float, default=0.1, help='Equivalent sample size')
    parser.add_argument('--n-search-ors', type=int, default=10, help='Number of OR candidates')
    parser.add_argument('--learn-clt', choices=['MI', 'BD'], default='MI', help='Method of learning CLTs')
    parser.add_argument('--n-iterations', type=int, default=20, help='Number of iterations for one EM')
    parser.add_argument('--patience', type=int, default=3,
                        help='Maximum number of iterations with no significant improvement')
    parser.add_argument('--store-all-models', action='store_true')
    parser.add_argument('--output', type=str, default='result/em/', help='Output store path')
    args = parser.parse_args()

    dataset = args.dataset
    ks = args.k
    ess = args.ess
    method_clt = args.learn_clt
    n_search_ors = args.n_search_ors
    n_iterations = args.n_iterations
    patience = args.patience
    is_store_all_models = args.store_all_models
    output_path = args.output

    time_dir = os.path.join(output_path, dataset + '_' + datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    os.makedirs(time_dir, exist_ok=True)
    ll_dir = os.path.join(time_dir, 'lls')
    os.makedirs(ll_dir, exist_ok=True)
    model_dir = os.path.join(time_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    csv_file = open(os.path.join(time_dir, dataset + '.csv'), 'w')
    fieldnames = ['dataset', 'n_components', 'val_ll', 'test_ll', 'time']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

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
    n_samples, n_features = train.shape

    print(dataset)

    print('{:<30}{}'.format('Training set size', train.shape))
    print('{:<30}{}'.format('Validation set size', valid.shape))
    print('{:<30}{}'.format('Test set size', test.shape))

    best_em_mean_val_ll = -np.inf
    best_k = None
    for k in ks:
        mcnet = MCNet(k=k)
        print()
        print('-' * 30 + ' k = ' + str(k) + '-' * 30)
        start = time.time()
        mean_iter_lls = mcnet.em(data=train,
                                 ess=ess,
                                 n_search_ors=n_search_ors,
                                 method_clt=method_clt,
                                 n_iterations=n_iterations,
                                 patience=patience)
        mcnet_time = time.time() - start

        np.savetxt(os.path.join(ll_dir, str(k) + '.iter.mean.lls'), mean_iter_lls)

        if is_store_all_models:
            with open(os.path.join(model_dir, str(k) + '_components.em'), 'wb') as em_file:
                pickle.dump(mcnet, em_file)

        mix_val_lls = []
        for i, cnet in enumerate(mcnet.cnets):
            val_lls = eval_cnet_log_likes(root=cnet.root, data=valid)
            mix_val_lls.append(val_lls)
        mix_val_lls = np.vstack(mix_val_lls)
        em_val_lls = logsumexp(mix_val_lls.T + np.log(mcnet.weights), axis=1)
        mean_em_val_ll = np.mean(em_val_lls)

        np.savetxt(os.path.join(ll_dir, str(k) + '.val.lls'), em_val_lls)

        if mean_em_val_ll > best_em_mean_val_ll:
            best_em_mean_val_ll = mean_em_val_ll
            best_k = k

        print('{:<30}{:.3f}'.format('MCNet time:', mcnet_time))
        print('{:<30}{:.3f}'.format('Mean Validation LL:', mean_em_val_ll))

        info = {'dataset': dataset,
                'n_components': k,
                'val_ll': '{:.3f}'.format(mean_em_val_ll),
                'time': '{:.3f}'.format(mcnet_time)}
        writer.writerow(info)
        csv_file.flush()

    train = np.vstack((train, valid))
    best_em = MCNet(k=best_k)
    print()
    print('-' * 30 + ' best k = ' + str(best_k) + '-' * 30)
    print('Re-train EM')
    start = time.time()
    mean_train_lls = best_em.em(data=train,
                                ess=ess,
                                n_search_ors=n_search_ors,
                                method_clt=method_clt,
                                n_iterations=n_iterations,
                                patience=patience)
    best_em_time = time.time() - start

    with open(os.path.join(time_dir, str(best_k) + '_components.best.em'), 'wb') as em_file:
        pickle.dump(best_em, em_file)

    mix_test_lls = []
    for i, cnet in enumerate(best_em.cnets):
        test_lls = eval_cnet_log_likes(root=cnet.root, data=test)
        mix_test_lls.append(test_lls)
    mix_test_lls = np.vstack(mix_test_lls)
    em_test_lls = logsumexp(mix_test_lls.T + np.log(best_em.weights), axis=1)
    mean_em_test_ll = np.mean(em_test_lls)

    print()
    print('{:<30}{:.3f}'.format('Mean test LL:', mean_em_test_ll))

    np.savetxt(os.path.join(ll_dir, 'best_' + str(best_k) + '_components.test.lls'), em_test_lls)

    info = {'dataset': dataset,
            'n_components': best_k,
            'test_ll': '{:.3f}'.format(mean_em_test_ll),
            'time': '{:.3f}'.format(best_em_time)}
    writer.writerow({})
    writer.writerow(info)
    csv_file.flush()
    csv_file.close()
