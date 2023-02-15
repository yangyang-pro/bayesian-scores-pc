import numpy as np

from ufal.chu_liu_edmonds import chu_liu_edmonds

from deeprob.spn.structure.cltree import BinaryCLT
from deeprob.utils.graph import maximum_spanning_tree

from stats import estimate_priors_joints_bayesian, compute_mutual_information
from score import compute_clt_bdeu_score_matrix


def breadth_first_order(predecessors):
    reversed_graph = {i: [] for i in range(len(predecessors))}
    root = 0
    for child, parent in enumerate(predecessors):
        if parent == -1:
            root = child
            continue
        reversed_graph[parent].append(child)
    bfs = []
    visited = [False] * len(predecessors)
    queue = [root]
    while queue:
        node = queue.pop(0)
        bfs.append(node)
        for child in reversed_graph[node]:
            if not visited[child]:
                queue.append(child)
                visited[child] = True
    return np.array(bfs)


def breath_first_search(graph, n_nodes, src, dst):
    visited = [False] * n_nodes
    visited[src] = True
    queue = [src]
    while queue:
        node = queue.pop(0)
        if node == dst:
            return True
        for child in graph[node]:
            if not visited[child]:
                queue.append(child)
                visited[child] = True
    return False


def check_path_length(graph, n_nodes, src, dst, max_len):
    visited = [False] * n_nodes
    queue = [(src, 0)]
    while queue:
        node, length = queue.pop(0)
        if node == dst:
            if length > max_len:
                return True
        for child in graph[node]:
            if not visited[child]:
                queue.append((child, length + 1))
                visited[child] = True
    return False


def estimate_clt_params_posterior(clt: BinaryCLT,
                                  data: np.ndarray,
                                  ess,
                                  fracs=None):
    n_samples, n_features = data.shape
    priors, joints = estimate_priors_joints_bayesian(data, ess=ess, fracs=fracs)

    vs = np.arange(n_features)
    params = np.einsum('ikl,il->ilk', joints[vs, clt.tree], np.reciprocal(priors[clt.tree]))
    params[clt.root] = priors[clt.root]

    # Re-normalize the factors, because there can be FP32 approximation errors
    params /= np.sum(params, axis=2, keepdims=True)
    return np.log(params)


# def learn_best_clt_mi(clt: BinaryCLT,
#                       data: np.ndarray,
#                       n_sample_root_indices=10,
#                       alpha=0.01,
#                       ess=1,
#                       clt_score_matrix=None,
#                       or_score_matrix=None,
#                       track=False):
#     n_samples, n_features = data.shape
#     if clt_score_matrix is None:
#         clt_score_matrix = compute_clt_bdeu_score_matrix(data=data, ess=ess)
#     if or_score_matrix is None:
#         or_score_matrix = compute_or_bdeu_score_matrix(data=data, ess=ess)
#
#     n_root_idx_candidates = min(data.shape[1], n_sample_root_indices)
#     root_idx_candidates = sorted(np.random.choice(n_features, n_root_idx_candidates, replace=False))
#     best_clt_score = -np.inf
#     best_clt = None
#     for i in root_idx_candidates:
#         clt.root = i
#         clt.tree = None
#         clt.fit(data=data, domain=[[0, 1]] * len(clt.scope), alpha=alpha)
#         clt_score = eval_tree_bdeu_score(clt.tree, clt_score_matrix, or_score_matrix)
#         if track:
#             print('CLT Root', i, 'Score', clt_score)
#         if clt_score > best_clt_score:
#             best_clt_score = clt_score
#             best_clt = clt
#     clt = best_clt
#     clt.root = clt.scope[clt.root]


def learn_clt_bdeu(clt: BinaryCLT,
                   data: np.ndarray,
                   root_idx=None,
                   ess=1,
                   clt_score_matrix=None):
    n_samples, n_features = data.shape
    if clt_score_matrix is None:
        clt_score_matrix = compute_clt_bdeu_score_matrix(data=data, ess=ess)
    clt_score_matrix = clt_score_matrix.astype(np.float64)

    if root_idx is None:
        root_idx = np.random.choice(n_features)
    clt_score_matrix[[0, root_idx]] = clt_score_matrix[[root_idx, 0]]
    clt_score_matrix[:, [0, root_idx]] = clt_score_matrix[:, [root_idx, 0]]
    heads, _ = chu_liu_edmonds(clt_score_matrix)
    heads = np.array(heads)
    root_indices = np.where(heads == root_idx)
    zero_indices = np.where(heads == 0)
    heads[root_indices], heads[zero_indices] = 0, root_idx
    heads[root_idx], heads[0] = heads[0], heads[root_idx]
    clt.root = root_idx
    clt.bfs = breadth_first_order(heads)
    clt.tree = heads


# def learn_best_clt_bdeu(clt: BinaryCLT,
#                         data: np.ndarray,
#                         n_sample_root_indices=10,
#                         alpha=0.01,
#                         ess=1,
#                         clt_score_matrix=None,
#                         or_score_matrix=None,
#                         track=False):
#     priors, joints = estimate_priors_joints(data, alpha=alpha)
#     if clt_score_matrix is None:
#         clt_score_matrix = compute_clt_bdeu_score_matrix(data=data, ess=ess / 2)
#     if or_score_matrix is None:
#         or_score_matrix = compute_or_bdeu_score_matrix(data=data, ess=ess)
#     clt_score_matrix = clt_score_matrix.astype(np.float64)
#
#     n_root_idx_candidates = min(data.shape[1], n_sample_root_indices)
#     mutual_info = compute_mutual_information(priors, joints)
#     root_idx_candidates = np.sum(mutual_info, axis=0).argpartition(-n_root_idx_candidates)[-n_root_idx_candidates:]
#
#     best_clt_score = -np.inf
#     best_root_idx = 0
#     best_heads = None
#     for i in root_idx_candidates:
#         reconstructed_score_matrix: np.ndarray = clt_score_matrix.copy()
#         reconstructed_score_matrix[[0, i]] = reconstructed_score_matrix[[i, 0]]
#         reconstructed_score_matrix[:, [0, i]] = reconstructed_score_matrix[:, [i, 0]]
#         heads, _ = chu_liu_edmonds(reconstructed_score_matrix)
#         heads = np.array(heads)
#         root_indices = np.where(heads == i)
#         zero_indices = np.where(heads == 0)
#         heads[root_indices], heads[zero_indices] = 0, i
#         heads[i], heads[0] = heads[0], heads[i]
#         clt_score = eval_tree_bdeu_score(heads, clt_score_matrix, or_score_matrix)
#         if track:
#             print('CLT Root', i, 'Score', clt_score)
#         if clt_score > best_clt_score:
#             best_clt_score = clt_score
#             best_root_idx = i
#             best_heads = heads
#     if track:
#         print('CLT Best root', best_root_idx, 'Best score', best_clt_score)
#     clt.root = best_root_idx
#     clt.bfs = breadth_first_order(best_heads)
#     clt.tree = best_heads
#     clt.params = np.log(clt.compute_clt_parameters(clt.bfs, clt.tree, priors, joints))


def learn_clt_mi(clt: BinaryCLT,
                 data: np.ndarray,
                 ess: float,
                 fracs: np.ndarray = None):
    _, n_features = data.shape
    # Choose a root variable randomly, if not specified
    if clt.root is None:
        clt.root = np.random.choice(n_features)

    # Estimate the priors and joints probabilities
    priors, joints = estimate_priors_joints_bayesian(data, ess=ess, fracs=fracs)

    if clt.tree is None:
        # Compute the mutual information
        mutual_info = compute_mutual_information(priors, joints)

        # Compute the CLT structure
        clt.bfs, clt.tree = maximum_spanning_tree(clt.root, mutual_info)
