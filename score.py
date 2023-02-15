import numpy as np
from scipy.special import gammaln

from stats import compute_prior_counts, compute_joint_counts


def compute_or_bdeu_score_matrix(data, ess, fracs=None):
    n_samples, n_features = data.shape
    if fracs is not None:
        n_samples = np.sum(fracs)
    prior_counts = compute_prior_counts(data=data, fracs=fracs)
    alpha_i = ess
    alpha_ik = ess / 2
    log_gamma_nodes = gammaln(alpha_i) - gammaln(n_samples + alpha_i) \
                      + np.sum(gammaln(prior_counts + alpha_ik) - gammaln(alpha_ik), axis=-1)
    return log_gamma_nodes


def compute_clt_bdeu_score_matrix(data, ess, fracs=None):
    joint_counts = compute_joint_counts(data=data, fracs=fracs)
    alpha_ij = ess / 2
    alpha_ijk = ess / (2 * 2)
    parent_counts = np.sum(joint_counts, axis=-2)
    log_gamma_pairs = gammaln(alpha_ij) - gammaln(parent_counts + alpha_ij) \
                      + np.sum(gammaln(joint_counts + alpha_ijk) - gammaln(alpha_ijk), axis=-2)
    return np.sum(log_gamma_pairs, axis=-1)
