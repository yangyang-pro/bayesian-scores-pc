import copy
import numpy as np

from nodes import ORNode
from score import compute_or_bdeu_score_matrix, compute_clt_bdeu_score_matrix


def breadth_first_search(root):
    bfs = []
    node_stack = [root]
    while node_stack:
        node: ORNode = node_stack.pop(0)
        bfs.append(node)
        if not node.is_leaf():
            for child in node.children:
                node_stack.append(child)
    return bfs


def eval_cnet_bdeu_score(root, data):
    nodes = breadth_first_search(root)
    cnet_score = 0.0
    for node in nodes:
        partition = data[node.row_indices][:, node.col_indices]
        or_score_matrix = compute_or_bdeu_score_matrix(partition, node.ess)
        if node.is_leaf():
            clt_score_matrix = compute_clt_bdeu_score_matrix(partition, node.ess / 2)
            cnet_score += eval_tree_bdeu_score(node.clt.tree, clt_score_matrix, or_score_matrix)
        else:
            cnet_score += or_score_matrix[node.scope.index(node.id)]
    return cnet_score


def eval_cnet_log_likes(root, data):
    n_samples, n_features = data.shape
    or_root = copy.copy(root)
    or_root.row_indices, or_root.col_indices = np.arange(n_samples), np.arange(n_features)
    node_stack = [or_root]
    cnet_log_likes = np.zeros(n_samples)
    while node_stack:
        node = node_stack.pop(0)
        partition = data[node.row_indices][:, node.col_indices]
        if node.is_leaf():
            cnet_log_likes[node.row_indices] += node.clt.log_likelihood(partition).squeeze()
            continue
        node_idx = node.scope.index(node.id)
        left_child = copy.copy(node.children[0])
        right_child = copy.copy(node.children[1])
        left_child.row_indices = node.row_indices[partition[:, node_idx] == 0]
        right_child.row_indices = node.row_indices[partition[:, node_idx] == 1]
        cnet_log_likes[left_child.row_indices] += np.log(node.weights[0])
        cnet_log_likes[right_child.row_indices] += np.log(node.weights[1])
        left_child.col_indices = np.delete(node.col_indices, obj=node_idx)
        right_child.col_indices = np.delete(node.col_indices, obj=node_idx)
        node_stack.append(left_child)
        node_stack.append(right_child)
    return cnet_log_likes


def eval_cnet_size(root):
    nodes = breadth_first_search(root)
    max_depth = -np.inf
    n_or_nodes = 0
    n_clts = 0
    n_params = 0
    for node in nodes:
        if node.is_leaf():
            n_clts += 1
            n_params += 2 * len(node.clt.scope) - 1
            if node.depth > max_depth:
                max_depth = node.depth
        else:
            n_or_nodes += 1
            n_params += 1
    return {'cnet_depth': max_depth, 'n_or': n_or_nodes, 'n_clts': n_clts, 'n_params': n_params}


def eval_tree_bdeu_score(tree, clt_score_matrix: np.ndarray, or_score_matrix: np.ndarray):
    root_idx = tree.argmin()
    parent_indices_no_root = np.delete(tree, obj=root_idx)
    child_indices_no_root = np.delete(np.arange(len(tree)), obj=root_idx)
    return np.sum(clt_score_matrix[child_indices_no_root, parent_indices_no_root]) + or_score_matrix[root_idx]


def eval_cnet_bic_score(root, data):
    nodes = breadth_first_search(root)
    cnet_score = 0
    for node in nodes:
        partition = data[node.row_indices][:, node.col_indices]
        if node.is_leaf():
            cnet_score += np.sum(node.clt.log_likelihood(partition)) \
                          - 0.5 * np.log(len(data)) * (2 * len(node.clt.scope) - 1)
        else:
            cnet_score += len(node.children[0].row_indices) * np.log(node.weights[0]) \
                          + len(node.children[1].row_indices) * np.log(node.weights[1]) \
                          - 0.5 * np.log(len(data))
    return cnet_score
