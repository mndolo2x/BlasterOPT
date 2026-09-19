"""
Joint Domain Adaptation (JDA) Submodule (`domain_adaptation`).
Aligns marginal P(X_src) ≈ P(X_tgt) and conditional P(Y|X_src) ≈ P(Y|X_tgt) distributions
using Maximum Mean Discrepancy (MMD) feature mapping.
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def compute_rbf_mmd(X_src: np.ndarray, X_tgt: np.ndarray, gamma: float = 1.0) -> float:
    """
    Computes Maximum Mean Discrepancy (MMD) distance with RBF kernel between source and target features.
    """
    K_ss = np.exp(-gamma * cdist(X_src, X_src, 'sqeuclidean'))
    K_tt = np.exp(-gamma * cdist(X_tgt, X_tgt, 'sqeuclidean'))
    K_st = np.exp(-gamma * cdist(X_src, X_tgt, 'sqeuclidean'))

    mmd = float(np.mean(K_ss) + np.mean(K_tt) - 2.0 * np.mean(K_st))
    return max(0.0, mmd)


class JDAAligner:
    """
    Joint Domain Adaptation (JDA) feature distribution aligner.
    Aligns marginal P(X) and conditional P(Y|X) distributions between source and target domains.
    """

    def __init__(self, n_components: int = 8, gamma: float = 1.0, reg: float = 1.0, n_classes: int = 4):
        self.n_components = n_components
        self.gamma = gamma
        self.reg = reg
        self.n_classes = n_classes
        self.projection_matrix: Optional[np.ndarray] = None

    def fit_transform(
        self,
        X_src: np.ndarray,
        X_tgt: np.ndarray,
        Y_src: Optional[np.ndarray] = None,
        Y_tgt_pseudo: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Fits JDA domain-invariant projection matrix aligning marginal and conditional MMD distributions.
        """
        n_src, d = X_src.shape
        n_tgt = X_tgt.shape[0]

        # Calculate kernel matrix K over combined samples
        X_combo = np.vstack([X_src, X_tgt])
        K = np.exp(-self.gamma * cdist(X_combo, X_combo, 'sqeuclidean'))

        # Marginal MMD matrix M0
        e_src = np.full((n_src, 1), 1.0 / n_src)
        e_tgt = np.full((n_tgt, 1), -1.0 / n_tgt)
        e = np.vstack([e_src, e_tgt])
        M0 = e @ e.T

        # Conditional MMD matrix Mc
        Mc = np.zeros((n_src + n_tgt, n_src + n_tgt))
        if Y_src is not None and Y_tgt_pseudo is not None:
            # Quantile bins for multi-output regression pseudo-labels
            ys_bin = np.digitize(Y_src[:, 0], np.linspace(np.min(Y_src[:, 0]), np.max(Y_src[:, 0]), self.n_classes))
            yt_bin = np.digitize(Y_tgt_pseudo[:, 0], np.linspace(np.min(Y_src[:, 0]), np.max(Y_src[:, 0]), self.n_classes))

            for c in range(1, self.n_classes + 1):
                idx_src_c = np.where(ys_bin == c)[0]
                idx_tgt_c = np.where(yt_bin == c)[0]

                ns_c = len(idx_src_c)
                nt_c = len(idx_tgt_c)

                if ns_c > 0 and nt_c > 0:
                    vec = np.zeros((n_src + n_tgt, 1))
                    vec[idx_src_c] = 1.0 / ns_c
                    vec[n_src + idx_tgt_c] = -1.0 / nt_c
                    Mc += vec @ vec.T

        # Total MMD matrix M = M0 + Mc
        M = M0 + Mc

        # Solve Rayleigh quotient eigenvalue problem
        n_total = n_src + n_tgt
        H = np.eye(n_total) - np.ones((n_total, n_total)) / n_total

        A = K @ M @ K + self.reg * np.eye(n_total)
        B = K @ H @ K

        eigvals, eigvecs = np.linalg.eigh(np.linalg.pinv(B) @ A)
        sorted_indices = np.argsort(eigvals)
        selected_indices = sorted_indices[:self.n_components]

        self.projection_matrix = eigvecs[:, selected_indices]

        # Transform features into aligned JDA subspace
        Z_combo = K @ self.projection_matrix
        Z_src = Z_combo[:n_src]
        Z_tgt = Z_combo[n_src:]

        # Compute MMD distance in original vs aligned subspace
        mmd_orig = compute_rbf_mmd(X_src, X_tgt, gamma=self.gamma)
        mmd_aligned = compute_rbf_mmd(Z_src, Z_tgt, gamma=self.gamma)

        return Z_src, Z_tgt, {
            "original_mmd": round(mmd_orig, 4),
            "aligned_mmd": round(mmd_aligned, 4),
            "mmd_reduction_pct": round(max(0.0, (1.0 - mmd_aligned / max(1e-5, mmd_orig)) * 100.0), 2)
        }

    def transform(self, X: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
        """Transforms new features into the fitted JDA aligned subspace."""
        if self.projection_matrix is None:
            return X
        K_new = np.exp(-self.gamma * cdist(X, X_ref, 'sqeuclidean'))
        if K_new.shape[1] == self.projection_matrix.shape[0]:
            return K_new @ self.projection_matrix
        return X[:, :self.n_components]
