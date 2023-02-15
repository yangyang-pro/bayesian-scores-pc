import numpy as np


def compute_prior_counts(data: np.ndarray, fracs: np.ndarray = None):
    # Compute the counts
    n_samples, n_features = data.shape
    if fracs is not None:
        n_samples = np.sum(fracs)
    counts_features = data.sum(axis=0) if fracs is None else (data.T * fracs).T.sum(axis=0)

    # Compute the prior counts
    prior_counts = np.empty(shape=(n_features, 2), dtype=np.float32)
    prior_counts[:, 1] = counts_features
    prior_counts[:, 0] = n_samples - prior_counts[:, 1]
    return prior_counts


def compute_joint_counts(data: np.ndarray, fracs: np.ndarray = None):
    # Compute the counts
    n_samples, n_features = data.shape
    if fracs is not None:
        n_samples = np.sum(fracs)
    counts_ones = np.dot(data.T, data) if fracs is None else np.dot(data.T, (data.T * fracs).T)
    counts_features = np.diag(counts_ones)
    counts_cols = counts_features * np.ones_like(counts_ones)
    counts_rows = np.transpose(counts_cols)

    # Compute the joint counts
    joint_counts = np.empty(shape=(n_features, n_features, 2, 2), dtype=np.float32)
    joint_counts[:, :, 0, 0] = n_samples - counts_cols - counts_rows + counts_ones
    joint_counts[:, :, 0, 1] = counts_cols - counts_ones
    joint_counts[:, :, 1, 0] = counts_rows - counts_ones
    joint_counts[:, :, 1, 1] = counts_ones
    return joint_counts


def estimate_priors_joints_bayesian(data: np.ndarray, ess: float, fracs: np.ndarray = None):
    if ess < 0.0:
        raise ValueError("The ESS must be non-negative")

    # Check the data dtype
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    # Compute the counts
    n_samples, n_features = data.shape
    if fracs is not None:
        n_samples = np.sum(fracs)
    counts_ones = np.dot(data.T, data).astype(np.float64) if fracs is None \
        else np.dot(data.T * fracs, data).astype(np.float64)
    counts_features = np.diag(counts_ones).astype(np.float64)
    counts_cols = (counts_features * np.ones_like(counts_ones)).astype(np.float64)
    counts_rows = np.transpose(counts_cols).astype(np.float64)

    # Compute the prior probabilities
    priors = np.empty(shape=(n_features, 2), dtype=np.float64)
    priors[:, 1] = (counts_features + ess / 2) / (n_samples + ess)
    priors[:, 0] = 1.0 - priors[:, 1]

    # Compute the joints probabilities
    joints = np.empty(shape=(n_features, n_features, 2, 2), dtype=np.float64)
    joints[:, :, 0, 0] = n_samples - counts_cols - counts_rows + counts_ones
    joints[:, :, 0, 1] = counts_cols - counts_ones
    joints[:, :, 1, 0] = counts_rows - counts_ones
    joints[:, :, 1, 1] = counts_ones
    joints = (joints + ess / 4) / (n_samples + ess)

    # Correct smoothing on the diagonal of joints array
    idx_features = np.arange(n_features)
    joints[idx_features, idx_features, 0, 0] = priors[:, 0]
    joints[idx_features, idx_features, 0, 1] = 0.0
    joints[idx_features, idx_features, 1, 0] = 0.0
    joints[idx_features, idx_features, 1, 1] = priors[:, 1]

    return priors, joints


def compute_mutual_information(priors: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """
    Compute the mutual information between each features, given priors and joints distributions.

    :param priors: The priors probability distributions, as a (N, D) Numpy array
                   having priors[i, k] = P(X_i=k).
    :param joints: The joints probability distributions, as a (N, N, D, D) Numpy array
                   having joints[i, j, k, l] = P(X_i=k, X_j=l).
    :return: The mutual information between each pair of features, as a (N, N) Numpy symmetric matrix.
    :raises ValueError: If there are inconsistencies between priors and joints arrays.
    :raises ValueError: If joints array is not symmetric.
    :raises ValueError: If priors or joints arrays don't encode valid probability distributions.
    """
    n_variables, n_values = priors.shape
    if joints.shape != (n_variables, n_variables, n_values, n_values):
        raise ValueError("There are inconsistencies between priors and joints distributions")
    if np.sum(joints - joints.transpose([1, 0, 3, 2])) != 0:
        raise ValueError("The joints probability distributions are expected to be symmetric")
    if not np.allclose(np.sum(priors, axis=1), 1.0):
        raise ValueError("The priors probability distributions are not valid")
    if not np.allclose(np.sum(joints, axis=(2, 3)), 1.0):
        raise ValueError("The joints probability distributions are not valid ")

    outers = np.multiply.outer(priors, priors).transpose([0, 2, 1, 3])
    # Ignore warnings of logarithm at zero (because NaNs on the diagonal will be zeroed later anyway)
    with np.errstate(divide='ignore', invalid='ignore'):
        mutual_info = np.sum(joints * (np.log(joints) - np.log(outers)), axis=(2, 3))
    np.fill_diagonal(mutual_info, 0.0)
    return mutual_info