import numpy as np
from typing import Union, List

from scipy.special import logsumexp
from sklearn.cluster import KMeans

from deeprob.spn.structure.leaf import Bernoulli
from deeprob.spn.structure.node import Sum, Product, assign_ids
from deeprob.spn.structure.cltree import BinaryCLT

from nodes import ORNode
from utils import eval_tree_bdeu_score, eval_cnet_log_likes
from stats import compute_prior_counts, compute_joint_counts
from score import compute_or_bdeu_score_matrix, compute_clt_bdeu_score_matrix
from learning import learn_clt_bdeu, estimate_clt_params_posterior, learn_clt_mi


class CNet:
    def __init__(self):
        self.root = None

    def bounded_search(self, data, ess, n_search_ors, method_clt, show_log=True):
        n_samples, n_features = data.shape
        clt = BinaryCLT(scope=list(range(n_features)))
        or_score_matrix = compute_or_bdeu_score_matrix(data=data, ess=ess)
        clt_score_matrix = compute_clt_bdeu_score_matrix(data=data, ess=ess)
        if method_clt == 'BD':
            learn_clt_bdeu(clt=clt, data=data, ess=ess, clt_score_matrix=clt_score_matrix)
        elif method_clt == 'MI':
            learn_clt_mi(clt=clt, data=data, ess=ess)
        else:
            raise NotImplementedError
        clt.params = estimate_clt_params_posterior(clt, data=data, ess=ess)
        clt_score = eval_tree_bdeu_score(clt.tree, clt_score_matrix, or_score_matrix)

        self.root = ORNode(scope=list(range(n_features)),
                           id=-1,
                           row_indices=np.arange(n_samples),
                           col_indices=np.arange(n_features),
                           ess=ess,
                           depth=0,
                           clt=clt,
                           clt_score=clt_score)

        node_stack = [self.root]
        while node_stack:
            node = node_stack.pop(0)
            if len(node.scope) == 1:
                continue

            partition = data[node.row_indices][:, node.col_indices]
            or_score_matrix = compute_or_bdeu_score_matrix(data=partition, ess=node.ess)

            k = min(n_search_ors, len(node.scope))
            search_indices, _ = self.__select_variable(partition, ess=node.ess, k=k)

            best_or_idx = -1
            best_cnet_score = -np.inf
            best_left_clt = None
            best_right_clt = None
            best_left_clt_score = -np.inf
            best_right_clt_score = -np.inf

            for i in search_indices:
                left_row_indices = node.row_indices[partition[:, i] == 0]
                right_row_indices = node.row_indices[partition[:, i] == 1]

                if len(left_row_indices) == 0 or len(right_row_indices) == 0:
                    continue

                child_col_indices = np.delete(node.col_indices, obj=i)
                left_partition = data[left_row_indices][:, child_col_indices]
                right_partition = data[right_row_indices][:, child_col_indices]
                new_scope = node.scope.copy()
                del new_scope[i]

                left_clt = BinaryCLT(scope=new_scope)
                left_or_score_matrix = compute_or_bdeu_score_matrix(data=left_partition, ess=node.ess / 2)
                left_clt_score_matrix = compute_clt_bdeu_score_matrix(data=left_partition, ess=node.ess / 2)

                right_clt = BinaryCLT(scope=new_scope)
                right_or_score_matrix = compute_or_bdeu_score_matrix(data=right_partition, ess=node.ess / 2)
                right_clt_score_matrix = compute_clt_bdeu_score_matrix(data=right_partition, ess=node.ess / 2)

                if method_clt == 'BD':
                    learn_clt_bdeu(clt=left_clt, data=left_partition,
                                   ess=node.ess / 2, clt_score_matrix=left_clt_score_matrix)
                    learn_clt_bdeu(clt=right_clt, data=right_partition,
                                   ess=node.ess / 2, clt_score_matrix=right_clt_score_matrix)
                elif method_clt == 'MI':
                    learn_clt_mi(clt=left_clt, data=left_partition, ess=node.ess / 2)
                    learn_clt_mi(clt=right_clt, data=right_partition, ess=node.ess / 2)
                else:
                    raise NotImplementedError

                left_clt.params = estimate_clt_params_posterior(left_clt, data=left_partition, ess=node.ess / 2)
                right_clt.params = estimate_clt_params_posterior(right_clt, data=right_partition, ess=node.ess / 2)

                left_clt_score = eval_tree_bdeu_score(tree=left_clt.tree,
                                                      clt_score_matrix=left_clt_score_matrix,
                                                      or_score_matrix=left_or_score_matrix)
                right_clt_score = eval_tree_bdeu_score(tree=right_clt.tree,
                                                       clt_score_matrix=right_clt_score_matrix,
                                                       or_score_matrix=right_or_score_matrix)
                cnet_score = left_clt_score + right_clt_score + or_score_matrix[i]

                if cnet_score > best_cnet_score:
                    best_cnet_score = cnet_score
                    best_or_idx = i
                    best_left_clt = left_clt
                    best_right_clt = right_clt
                    best_left_clt_score = left_clt_score
                    best_right_clt_score = right_clt_score

            if show_log:
                print('{:<20}{:<15}'
                      '{:<20}{:<15}'.format('Depth:', node.depth,
                                            'ESS', node.ess))
                print('{:<20}{:<15.3f}'
                      '{:<20}{:<15.3f}'
                      '{:<20}{:<15.3f}'.format('Left CLT score:', best_left_clt_score,
                                               'Right CLT score:', best_right_clt_score,
                                               'OR score:', or_score_matrix[best_or_idx]))

                print('{:<20}{:<15}'
                      '{:<20}{:<15.3f}'.format('Best OR idx:', best_or_idx,
                                               'CNet score:', best_cnet_score))
                print('{:<20}{:<15}'
                      '{:<20}{:<15.3f}'.format('CLT root idx:', node.clt.root,
                                               'CLT score:', node.clt_score))

            if best_cnet_score > node.clt_score:
                node.id = node.scope[best_or_idx]
                node.clt = None
                left_row_indices = node.row_indices[partition[:, best_or_idx] == 0]
                right_row_indices = node.row_indices[partition[:, best_or_idx] == 1]
                col_indices = np.delete(node.col_indices, obj=best_or_idx)
                if show_log:
                    print('+' * 30 + ' Cut ' + '+' * 30)
                    print('{:<20}{:<15}'
                          '{:<20}{:<15}'.format('Left samples:', len(left_row_indices),
                                                'Right samples:', len(right_row_indices)))
                    print('{:<20}{:<15}'.format('No. features:', len(col_indices)))
                left_weight = (len(left_row_indices) + node.ess / 2) / (len(node.row_indices) + node.ess)
                right_weight = 1 - left_weight
                new_scope = node.scope.copy()
                del new_scope[best_or_idx]
                left_child = ORNode(scope=new_scope,
                                    id=node.id,
                                    row_indices=left_row_indices,
                                    col_indices=col_indices,
                                    ess=node.ess / 2,
                                    clt=best_left_clt,
                                    clt_score=best_left_clt_score,
                                    depth=node.depth + 1,
                                    flag=0)
                right_child = ORNode(scope=new_scope,
                                     id=node.id,
                                     row_indices=right_row_indices,
                                     col_indices=col_indices,
                                     ess=node.ess / 2,
                                     clt=best_right_clt,
                                     clt_score=best_right_clt_score,
                                     depth=node.depth + 1,
                                     flag=1)
                node_stack.append(left_child)
                node_stack.append(right_child)
                node.weights = [left_weight, right_weight]
                node.children = [left_child, right_child]

                # mean_marginal_ll = eval_cnet_bdeu_score(root=self.root,
                #                                         data=data,
                #                                         ess_or=ess_or,
                #                                         ess_clt=ess_clt) / len(data)
                # mean_train_ll = eval_cnet_log_likelihood(root=self.root, data=data) / len(data)
                # mean_test_ll = eval_cnet_log_likelihood(root=self.root, data=test_data) / len(test_data)
                # print('{:<20}{:<6}'.format('Mean marginal LL:', mean_marginal_ll))
                # print('{:<20}{:<6}'.format('Mean training LL:', mean_train_ll))
                # print('{:<20}{:<6}'.format('Mean test LL:', mean_test_ll))
                # mean_marginal_lls.append(mean_marginal_ll)
                # mean_train_lls.append(mean_train_ll)
                # mean_test_lls.append(mean_test_ll)
            else:
                if show_log:
                    print('*' * 30 + ' CLT ' + '*' * 30)
                    print('{:<20}{:<15}'
                          '{:<20}{:<15}'.format('No. samples:', partition.shape[0],
                                                'No. features:', partition.shape[1]))
            if show_log:
                print()
                print('-' * 70)
                print()
        # plt.plot(np.arange(len(mean_marginal_lls)), mean_marginal_lls, label='Marginal LL')
        # plt.plot(np.arange(len(mean_train_lls)), mean_train_lls, label='Training LL')
        # plt.plot(np.arange(len(mean_test_lls)), mean_test_lls, label='Test LL')
        # plt.legend()
        # plt.show()

    def bounded_search_frac(self, data, ess, n_search_ors, method_clt, fracs, show_log=True):
        n_samples, n_features = data.shape
        clt = BinaryCLT(scope=list(range(n_features)))
        or_score_matrix = compute_or_bdeu_score_matrix(data=data, ess=ess, fracs=fracs)
        clt_score_matrix = compute_clt_bdeu_score_matrix(data=data, ess=ess, fracs=fracs)
        if method_clt == 'MI':
            learn_clt_mi(clt=clt, data=data, ess=ess, fracs=fracs)
        else:
            raise NotImplementedError
        clt.params = estimate_clt_params_posterior(clt, data=data, ess=ess, fracs=fracs)
        clt_score = eval_tree_bdeu_score(clt.tree, clt_score_matrix, or_score_matrix)

        self.root = ORNode(scope=list(range(n_features)),
                           id=-1,
                           row_indices=np.arange(n_samples),
                           col_indices=np.arange(n_features),
                           ess=ess,
                           depth=0,
                           clt=clt,
                           clt_score=clt_score)

        node_stack = [self.root]
        while node_stack:
            node = node_stack.pop(0)
            if len(node.scope) == 1:
                continue

            partition = data[node.row_indices][:, node.col_indices]
            node_fracs = fracs[node.row_indices]
            or_score_matrix = compute_or_bdeu_score_matrix(data=partition, ess=node.ess, fracs=node_fracs)

            k = min(n_search_ors, len(node.scope))
            search_indices, _ = self.__select_variable(data=partition, k=k, fracs=node_fracs, ess=node.ess)

            best_or_idx = -1
            best_cnet_score = -np.inf
            best_left_clt = None
            best_right_clt = None
            best_left_clt_score = -np.inf
            best_right_clt_score = -np.inf

            for i in search_indices:
                left_row_indices = node.row_indices[partition[:, i] == 0]
                right_row_indices = node.row_indices[partition[:, i] == 1]

                if len(left_row_indices) == 0 or len(right_row_indices) == 0:
                    continue

                left_fracs, right_fracs = fracs[left_row_indices], fracs[right_row_indices]
                if np.sum(left_fracs) == 0 or np.sum(right_fracs) == 0:
                    continue

                child_col_indices = np.delete(node.col_indices, obj=i)
                left_partition = data[left_row_indices][:, child_col_indices]
                right_partition = data[right_row_indices][:, child_col_indices]
                new_scope = node.scope.copy()
                del new_scope[i]

                left_clt = BinaryCLT(scope=new_scope)
                left_or_score_matrix = compute_or_bdeu_score_matrix(data=left_partition,
                                                                    ess=node.ess / 2,
                                                                    fracs=left_fracs)
                left_clt_score_matrix = compute_clt_bdeu_score_matrix(data=left_partition,
                                                                      ess=node.ess / 2,
                                                                      fracs=left_fracs)

                right_clt = BinaryCLT(scope=new_scope)
                right_or_score_matrix = compute_or_bdeu_score_matrix(data=right_partition,
                                                                     ess=node.ess / 2,
                                                                     fracs=right_fracs)
                right_clt_score_matrix = compute_clt_bdeu_score_matrix(data=right_partition,
                                                                       ess=node.ess / 2,
                                                                       fracs=right_fracs)

                if method_clt == 'MI':
                    learn_clt_mi(clt=left_clt, data=left_partition, fracs=left_fracs, ess=node.ess / 2)
                    learn_clt_mi(clt=right_clt, data=right_partition, fracs=right_fracs, ess=node.ess / 2)
                else:
                    raise NotImplementedError

                left_clt.params = estimate_clt_params_posterior(clt=left_clt,
                                                                data=left_partition,
                                                                ess=node.ess / 2,
                                                                fracs=left_fracs)
                right_clt.params = estimate_clt_params_posterior(clt=right_clt,
                                                                 data=right_partition,
                                                                 ess=node.ess / 2,
                                                                 fracs=right_fracs)

                left_clt_score = eval_tree_bdeu_score(tree=left_clt.tree,
                                                      clt_score_matrix=left_clt_score_matrix,
                                                      or_score_matrix=left_or_score_matrix)
                right_clt_score = eval_tree_bdeu_score(tree=right_clt.tree,
                                                       clt_score_matrix=right_clt_score_matrix,
                                                       or_score_matrix=right_or_score_matrix)
                cnet_score = left_clt_score + right_clt_score + or_score_matrix[i]

                if cnet_score > best_cnet_score:
                    best_cnet_score = cnet_score
                    best_or_idx = i
                    best_left_clt = left_clt
                    best_right_clt = right_clt
                    best_left_clt_score = left_clt_score
                    best_right_clt_score = right_clt_score

            if show_log:
                print('{:<20}{:<15}'
                      '{:<20}{:<15}'.format('Depth:', node.depth,
                                            'ESS', node.ess))
                print('{:<20}{:<15.3f}'
                      '{:<20}{:<15.3f}'
                      '{:<20}{:<15.3f}'.format('Left CLT score:', best_left_clt_score,
                                               'Right CLT score:', best_right_clt_score,
                                               'OR score:', or_score_matrix[best_or_idx]))

                print('{:<20}{:<15}'
                      '{:<20}{:<15.3f}'.format('Best OR idx:', best_or_idx,
                                               'CNet score:', best_cnet_score))
                print('{:<20}{:<15}'
                      '{:<20}{:<15.3f}'.format('CLT root idx:', node.clt.root,
                                               'CLT score:', node.clt_score))

            if best_cnet_score > node.clt_score:
                node.id = node.scope[best_or_idx]
                node.clt = None
                left_row_indices = node.row_indices[partition[:, best_or_idx] == 0]
                right_row_indices = node.row_indices[partition[:, best_or_idx] == 1]
                col_indices = np.delete(node.col_indices, obj=best_or_idx)
                if show_log:
                    print('+' * 30 + ' Cut ' + '+' * 30)
                    print('{:<20}{:<15}'
                          '{:<20}{:<15}'.format('Left samples:', len(left_row_indices),
                                                'Right samples:', len(right_row_indices)))
                    print('{:<20}{:<15}'.format('No. features:', len(col_indices)))
                node_fracs = fracs[node.row_indices]
                left_fracs, right_fracs = fracs[left_row_indices], fracs[right_row_indices]
                left_weight = (np.sum(left_fracs) + node.ess / 2) / (np.sum(node_fracs) + node.ess)
                right_weight = 1 - left_weight
                new_scope = node.scope.copy()
                del new_scope[best_or_idx]
                left_child = ORNode(scope=new_scope,
                                    id=node.id,
                                    row_indices=left_row_indices,
                                    col_indices=col_indices,
                                    ess=node.ess / 2,
                                    clt=best_left_clt,
                                    clt_score=best_left_clt_score,
                                    depth=node.depth + 1,
                                    flag=0)
                right_child = ORNode(scope=new_scope,
                                     id=node.id,
                                     row_indices=right_row_indices,
                                     col_indices=col_indices,
                                     ess=node.ess / 2,
                                     clt=best_right_clt,
                                     clt_score=best_right_clt_score,
                                     depth=node.depth + 1,
                                     flag=1)
                node_stack.append(left_child)
                node_stack.append(right_child)
                node.weights = [left_weight, right_weight]
                node.children = [left_child, right_child]
            else:
                if show_log:
                    print('*' * 30 + ' CLT ' + '*' * 30)
                    print('{:<20}{:<15}'
                          '{:<20}{:<15}'.format('No. samples:', partition.shape[0],
                                                'No. features:', partition.shape[1]))
            if show_log:
                print()
                print('-' * 70)
                print()

    def bounded_search_bic(self, data, n_search_ors, alpha=0.01):
        n_samples, n_features = data.shape
        clt = BinaryCLT(scope=list(range(n_features)))
        clt.fit(data=data, domain=[[0, 1]] * n_features, alpha=alpha)
        clt_score = np.sum(clt.log_likelihood(data)) - 0.5 * np.log(n_samples) * (2 * n_features - 1)

        self.root = ORNode(scope=list(range(n_features)),
                           id=-1,
                           row_indices=np.arange(n_samples),
                           col_indices=np.arange(n_features),
                           depth=0,
                           clt=clt,
                           clt_score=clt_score)

        node_stack = [self.root]
        while node_stack:
            node = node_stack.pop(0)
            if len(node.scope) == 1:
                continue

            partition = data[node.row_indices][:, node.col_indices]

            k = min(n_search_ors, len(node.scope))
            search_indices, _ = self.__select_variable(partition, k=k, alpha=alpha, ess=None)

            best_or_idx = -1
            best_cnet_score = -np.inf
            best_left_clt = None
            best_right_clt = None
            best_left_clt_score = 0.0
            best_right_clt_score = 0.0
            for i in search_indices:
                left_row_indices = node.row_indices[partition[:, i] == 0]
                right_row_indices = node.row_indices[partition[:, i] == 1]

                if len(left_row_indices) == 0 or len(right_row_indices) == 0:
                    continue

                child_col_indices = np.delete(node.col_indices, obj=i)
                left_partition = data[left_row_indices][:, child_col_indices]
                right_partition = data[right_row_indices][:, child_col_indices]
                new_scope = node.scope.copy()
                del new_scope[i]

                left_weight = (len(left_row_indices) + alpha) / (len(node.row_indices) + 2 * alpha)
                right_weight = 1 - left_weight

                left_clt = BinaryCLT(scope=new_scope)
                right_clt = BinaryCLT(scope=new_scope)

                left_clt.fit(data=left_partition, domain=[[0, 1]] * len(new_scope), alpha=alpha)
                right_clt.fit(data=right_partition, domain=[[0, 1]] * len(new_scope), alpha=alpha)

                left_clt_score = np.sum(left_clt.log_likelihood(left_partition)) - \
                                 0.5 * np.log(len(data)) * (2 * len(new_scope) - 1)
                right_clt_score = np.sum(right_clt.log_likelihood(right_partition)) - \
                                  0.5 * np.log(len(data)) * (2 * len(new_scope) - 1)
                or_score = len(left_partition) * np.log(left_weight) + len(right_partition) * np.log(right_weight) - \
                           0.5 * np.log(len(data))
                cnet_score = left_clt_score + right_clt_score + or_score

                if cnet_score > best_cnet_score:
                    best_cnet_score = cnet_score
                    best_or_idx = i
                    best_left_clt = left_clt
                    best_right_clt = right_clt
                    best_left_clt_score = left_clt_score
                    best_right_clt_score = right_clt_score

            print('{:<20}{:<6}'.format('Depth:', node.depth))
            print('{:<20}{:<6}  {:<20}{:11.3f}'.format('Best OR idx:',
                                                       best_or_idx,
                                                       'CNet score:',
                                                       best_cnet_score))
            print('{:<20}{:<6}  {:<20}{:11.3f}'.format('CLT root idx:',
                                                       node.clt.root,
                                                       'CLT score:',
                                                       node.clt_score))

            if best_cnet_score > node.clt_score:
                node.id = node.scope[best_or_idx]
                node.clt = None
                left_row_indices = node.row_indices[partition[:, best_or_idx] == 0]
                right_row_indices = node.row_indices[partition[:, best_or_idx] == 1]
                col_indices = np.delete(node.col_indices, obj=best_or_idx)
                print('+' * 27 + ' Cut ' + '+' * 28)
                print('{:<20}{:<6}  {:<20}{:11}'.format('Left samples:',
                                                        len(left_row_indices),
                                                        'Right samples:',
                                                        len(right_row_indices)))
                print('{:<20}{:<6}'.format('No. features:',
                                           len(col_indices)))

                left_weight = (len(left_row_indices) + alpha) / (len(node.row_indices) + 2 * alpha)
                right_weight = 1 - left_weight
                new_scope = node.scope.copy()
                del new_scope[best_or_idx]
                left_child = ORNode(scope=new_scope,
                                    id=node.id,
                                    row_indices=left_row_indices,
                                    col_indices=col_indices,
                                    clt=best_left_clt,
                                    clt_score=best_left_clt_score,
                                    depth=node.depth + 1,
                                    flag=0)
                right_child = ORNode(scope=new_scope,
                                     id=node.id,
                                     row_indices=right_row_indices,
                                     col_indices=col_indices,
                                     clt=best_right_clt,
                                     clt_score=best_right_clt_score,
                                     depth=node.depth + 1,
                                     flag=1)
                node_stack.append(left_child)
                node_stack.append(right_child)
                node.weights = [left_weight, right_weight]
                node.children = [left_child, right_child]
            else:
                print('*' * 27 + ' CLT ' + '*' * 28)
                print('{:<20}{:<6}  {:<20}{:11}'.format('No. samples:',
                                                        partition.shape[0],
                                                        'No. features:',
                                                        partition.shape[1]))
            print()
            print('-' * 60)
            print()

    def learn_cnet_entropy(self, data, alpha, min_n_samples, min_n_features):
        n_samples, n_features = data.shape
        self.root = ORNode(scope=list(range(n_features)),
                           row_indices=np.arange(n_samples),
                           col_indices=np.arange(n_features),
                           depth=0)
        node_stack = [self.root]
        while node_stack:
            node = node_stack.pop(0)
            partition = data[node.row_indices][:, node.col_indices]
            n_samples, n_features = partition.shape
            if n_samples <= min_n_samples or n_features <= min_n_features:
                # print('stopped due to few samples')
                clt = BinaryCLT(scope=node.scope)
                clt.fit(data=partition, domain=[[0, 1]] * n_features, alpha=alpha)
                node.clt = clt
                continue
            best_or_idx, mean_entropy, max_info_gain = self.__select_variable_entropy(partition, alpha=alpha)
            if mean_entropy < 0.01 or max_info_gain <= 0:
                # print('stopped due to small entropy')
                clt = BinaryCLT(scope=node.scope)
                clt.fit(data=partition, domain=[[0, 1]] * n_features, alpha=alpha)
                node.clt = clt
                continue
            left_row_indices = node.row_indices[partition[:, best_or_idx] == 0]
            right_row_indices = node.row_indices[partition[:, best_or_idx] == 1]
            left_col_indices = right_col_indices = np.delete(node.col_indices, obj=best_or_idx)
            left_weight = (len(left_row_indices) + alpha) / (len(node.row_indices) + 2 * alpha)
            right_weight = 1 - left_weight
            new_scope = node.scope.copy()
            del new_scope[best_or_idx]
            left_child = ORNode(scope=new_scope,
                                row_indices=left_row_indices,
                                col_indices=left_col_indices,
                                depth=node.depth + 1,
                                flag=0)
            right_child = ORNode(scope=new_scope,
                                 row_indices=right_row_indices,
                                 col_indices=right_col_indices,
                                 depth=node.depth + 1,
                                 flag=1)
            node_stack.append(left_child)
            node_stack.append(right_child)
            node.children = [left_child, right_child]
            node.weights = [left_weight, right_weight]
            node.id = node.scope[best_or_idx]

    def learn_xcnet(self, data, alpha, min_n_samples, min_n_features):
        n_samples, n_features = data.shape
        self.root = ORNode(scope=list(range(n_features)),
                           row_indices=np.arange(n_samples),
                           col_indices=np.arange(n_features),
                           depth=0)
        node_stack = [self.root]
        while node_stack:
            node = node_stack.pop(0)
            partition = data[node.row_indices][:, node.col_indices]
            n_samples, n_features = partition.shape
            if n_samples <= min_n_samples or n_features <= min_n_features:
                clt = BinaryCLT(scope=node.scope)
                clt.fit(data=partition, domain=[[0, 1]] * n_features, alpha=alpha)
                node.clt = clt
                continue
            or_idx = np.random.choice(n_features)
            left_row_indices = node.row_indices[partition[:, or_idx] == 0]
            right_row_indices = node.row_indices[partition[:, or_idx] == 1]
            left_col_indices = right_col_indices = np.delete(node.col_indices, obj=or_idx)
            left_weight = (len(left_row_indices) + alpha) / (len(node.row_indices) + 2 * alpha)
            right_weight = 1 - left_weight
            new_scope = node.scope.copy()
            del new_scope[or_idx]
            left_child = ORNode(scope=new_scope,
                                row_indices=left_row_indices,
                                col_indices=left_col_indices,
                                depth=node.depth + 1,
                                flag=0)
            right_child = ORNode(scope=new_scope,
                                 row_indices=right_row_indices,
                                 col_indices=right_col_indices,
                                 depth=node.depth + 1,
                                 flag=1)
            node_stack.append(left_child)
            node_stack.append(right_child)
            node.children = [left_child, right_child]
            node.weights = [left_weight, right_weight]
            node.id = node.scope[or_idx]

    @staticmethod
    def __select_variable_entropy(data, alpha=1.0):
        n_samples, n_features = data.shape
        counts_features = np.sum(data, axis=0)

        prior_counts = compute_prior_counts(data)
        joint_counts = compute_joint_counts(data)
        priors = (prior_counts + 2 * alpha) / (n_samples + 4 * alpha)
        priors[:, 0] = 1.0 - priors[:, 1]
        mean_entropy = -(priors * np.log(priors)).sum() / n_features

        conditionals = np.empty((n_features, n_features, 2, 2), dtype=np.float32)
        # as we are computing the probabilities for all nodes after cutting on a node, the laplace smoothing factor is
        # essentially the same as computing general prior probabilities
        conditionals[:, :, 0, 0] = ((joint_counts[:, :, 0, 0] + 2 * alpha).T / (prior_counts[:, 0] + 4 * alpha)).T
        conditionals[:, :, 0, 1] = ((joint_counts[:, :, 0, 1] + 2 * alpha).T / (prior_counts[:, 0] + 4 * alpha)).T
        conditionals[:, :, 1, 0] = ((joint_counts[:, :, 1, 0] + 2 * alpha).T / (prior_counts[:, 1] + 4 * alpha)).T
        conditionals[:, :, 1, 1] = ((joint_counts[:, :, 1, 1] + 2 * alpha).T / (prior_counts[:, 1] + 4 * alpha)).T

        vs = np.repeat(np.arange(n_features)[None, :], n_features, axis=0)
        vs = vs[~np.eye(vs.shape[0], dtype=bool)].reshape(vs.shape[0], -1)
        parents = np.repeat(np.arange(n_features)[:, None], n_features - 1, axis=1)

        ratio_features = counts_features / n_samples
        entropies = ratio_features * \
                    np.mean(-np.sum(conditionals[parents, vs, 1, :] * np.log(conditionals[parents, vs, 1, :]), axis=-1),
                            axis=1) + \
                    (1 - ratio_features) * \
                    np.mean(-np.sum(conditionals[parents, vs, 0, :] * np.log(conditionals[parents, vs, 0, :]), axis=-1),
                            axis=1)

        info_gains = mean_entropy - entropies
        selected_idx = np.argmax(info_gains)
        return selected_idx, mean_entropy, info_gains[selected_idx]

    @staticmethod
    def __select_variable(data, k=1, alpha=0.01, ess=None, fracs=None):
        # Compute the counts
        n_samples, n_features = data.shape
        if fracs is not None:
            n_samples = np.sum(fracs)
        counts_features = data.sum(axis=0) if fracs is None else (data.T * fracs).T.sum(axis=0)

        prior_counts = compute_prior_counts(data, fracs=fracs)
        joint_counts = compute_joint_counts(data, fracs=fracs)
        smoothing_joint, smoothing_prior = 2 * alpha, 4 * alpha
        if ess is not None:
            smoothing_joint, smoothing_prior = ess / 2, ess
            if ess < 0.01:
                prior_counts = prior_counts.astype(np.float64)
                joint_counts = joint_counts.astype(np.float64)
        log_priors = np.log(prior_counts + smoothing_joint) - np.log(n_samples + smoothing_prior)
        mean_entropy = -(log_priors * np.exp(log_priors)).sum() / n_features

        conditionals = np.empty((n_features, n_features, 2, 2), dtype=prior_counts.dtype)
        conditionals[:, :, 0, 0] = ((joint_counts[:, :, 0, 0] + smoothing_joint).T /
                                    (prior_counts[:, 0] + smoothing_prior)).T
        conditionals[:, :, 0, 1] = ((joint_counts[:, :, 0, 1] + smoothing_joint).T /
                                    (prior_counts[:, 0] + smoothing_prior)).T
        conditionals[:, :, 1, 0] = ((joint_counts[:, :, 1, 0] + smoothing_joint).T /
                                    (prior_counts[:, 1] + smoothing_prior)).T
        conditionals[:, :, 1, 1] = ((joint_counts[:, :, 1, 1] + smoothing_joint).T /
                                    (prior_counts[:, 1] + smoothing_prior)).T

        vs = np.repeat(np.arange(n_features)[None, :], n_features, axis=0)
        vs = vs[~np.eye(vs.shape[0], dtype=bool)].reshape(vs.shape[0], -1)
        parents = np.repeat(np.arange(n_features)[:, None], n_features - 1, axis=1)

        ratio_features = counts_features / n_samples
        entropies = ratio_features * \
                    np.mean(-np.sum(conditionals[parents, vs, 1, :] * np.log(conditionals[parents, vs, 1, :]), axis=-1),
                            axis=1) + \
                    (1 - ratio_features) * \
                    np.mean(-np.sum(conditionals[parents, vs, 0, :] * np.log(conditionals[parents, vs, 0, :]), axis=-1),
                            axis=1)

        info_gains = mean_entropy - entropies
        selected_idx = np.argmax(info_gains) if k == 1 else np.argpartition(info_gains, -k)[-k:]
        return selected_idx, mean_entropy

    def to_pc(self):
        # Post-Order exploration
        neg_buffer, pos_buffer = [], []
        nodes_stack = [self.root]
        last_node_visited = None
        while nodes_stack:
            node = nodes_stack[-1]
            if node.is_leaf() or (last_node_visited in node.children):
                # print(node.id)
                leaves: List[Union[Bernoulli, Sum]] = [
                    Bernoulli(node.id, p=0.0),
                    Bernoulli(node.id, p=1.0)
                ]
                if not node.is_leaf():
                    neg_prod = Product(children=[leaves[0], neg_buffer[-1]])
                    pos_prod = Product(children=[leaves[1], pos_buffer[-1]])
                    del neg_buffer[-1]
                    del pos_buffer[-1]
                    sum_children = [neg_prod, pos_prod]
                    weights = node.weights
                    if node.flag == 0:
                        neg_buffer.append(Sum(children=sum_children, weights=weights))
                    else:
                        pos_buffer.append(Sum(children=sum_children, weights=weights))
                else:
                    node_pc = node.clt.to_pc()
                    if node.flag == 0:
                        neg_buffer.append(node_pc)
                    else:
                        pos_buffer.append(node_pc)
                last_node_visited = nodes_stack.pop()
            else:
                nodes_stack.extend(node.children)
        pc = pos_buffer[0]
        return assign_ids(pc)

    def sample(self, n_samples):
        root: ORNode = self.root
        x = np.zeros((n_samples, len(root.scope)))
        for i in range(n_samples):
            node = root
            while not node.is_leaf():
                or_id = node.id
                weights = node.weights
                children = node.children
                sample = np.random.binomial(1, weights[1], 1)[0]
                x[i, or_id] = sample
                node = children[sample]
            sample_clt = node.clt.sample(x=np.array([np.nan] * len(node.scope))[np.newaxis, :])
            x[i, node.scope] = sample_clt
        return x


class MCNet:
    def __init__(self, k):
        self.k = k
        self.cnets = [CNet() for _ in range(k)]
        self.weights = np.array([1 / k for _ in range(k)])

    def em(self, data, ess, n_search_ors, method_clt, n_iterations, stopping_threshold=0.01, patience=3):
        n_samples, n_features = data.shape
        k_means = KMeans(n_clusters=self.k)
        assignments = k_means.fit_predict(data)
        cluster_indices = [np.where(assignments == i)[0] for i in np.unique(assignments)]
        for i, cnet in enumerate(self.cnets):
            partition = data[cluster_indices[i]]
            cnet.bounded_search(data=partition,
                                ess=ess,
                                n_search_ors=n_search_ors,
                                method_clt=method_clt,
                                show_log=False)
        mix_log_likes = []
        for i, cnet in enumerate(self.cnets):
            log_likes = eval_cnet_log_likes(root=cnet.root, data=data)
            mix_log_likes.append(log_likes)
        mix_log_likes = np.vstack(mix_log_likes)
        log_likes = logsumexp(mix_log_likes.T + np.log(self.weights), axis=1)

        print('{:<30}{}'.format('Initial weights:', self.weights))
        print('{:<30}{:.3f}'.format('Mean train LL:', np.mean(log_likes)))

        mean_lls = [np.mean(log_likes)]
        n_stags = 0
        for iter in range(n_iterations):
            # E step
            mix_fracs = []
            for i, cnet in enumerate(self.cnets):
                log_likes = eval_cnet_log_likes(cnet.root, data)
                mix_fracs.append(log_likes + np.log(self.weights[i]))
            mix_fracs = np.vstack(mix_fracs)
            sum_fracs = logsumexp(mix_fracs, axis=0)
            mix_fracs -= sum_fracs
            mix_fracs = np.exp(mix_fracs)

            # M step
            self.weights = np.sum(mix_fracs, axis=1) / n_samples
            for i, cnet in enumerate(self.cnets):
                cnet.bounded_search_frac(data=data,
                                         ess=ess,
                                         n_search_ors=n_search_ors,
                                         method_clt=method_clt,
                                         fracs=mix_fracs[i],
                                         show_log=False)

            # check convergence
            mix_log_likes = []
            for i, cnet in enumerate(self.cnets):
                log_likes = eval_cnet_log_likes(root=cnet.root, data=data)
                mix_log_likes.append(log_likes)
            mix_log_likes = np.vstack(mix_log_likes)
            log_likes = logsumexp(mix_log_likes.T + np.log(self.weights), axis=1)

            mean_ll = np.mean(log_likes)
            print('{:<30}{}'.format('Current weights:', self.weights))
            print('{:<30}{:.3f}'.format('Mean train LL:', mean_ll))
            mean_lls.append(mean_ll)
            if mean_lls[-1] - mean_lls[-2] <= -stopping_threshold * mean_lls[-2]:
                n_stags += 1
                if n_stags >= patience:
                    break
            else:
                n_stags = 0
        return mean_lls
