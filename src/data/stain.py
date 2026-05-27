"""Stain normalization for H&E histopathology images.

Three methods supported (Reinhard 2001, Macenko 2009, Vahadane 2016).
Each fitter follows the same scikit-learn-style API:

    fitter = get_fitter("macenko").fit(reference_image_rgb_uint8)
    normalized_image_rgb_uint8 = fitter.transform(source_image_rgb_uint8)

All three normalize a source image's stain appearance to match a fixed
reference image. Inputs and outputs are HxWx3 uint8 RGB arrays. The
underlying math is in optical-density (OD = -log((rgb + 1) / 256))
space for Macenko and Vahadane; Reinhard works in CIE LAB space.

Implementation choices:
  - Pure NumPy + scikit-image + scikit-learn. No staintools dependency
    (license concerns noted in the project risk register).
  - Vahadane uses sklearn.decomposition.NMF with sparsity regularisation
    in place of the canonical SPAMS-based SNMF.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import NMF
from skimage.color import rgb2lab, lab2rgb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rgb_to_od(img: np.ndarray) -> np.ndarray:
    """Convert HxWx3 uint8 RGB to N×3 optical density."""
    img = img.astype(np.float64)
    return -np.log((img + 1.0) / 256.0)


def _od_to_rgb(od_flat: np.ndarray, shape: tuple) -> np.ndarray:
    """Convert N×3 optical density back to HxWx3 uint8 RGB."""
    rgb = 256.0 * np.exp(-od_flat) - 1.0
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb.reshape(shape)


# ---------------------------------------------------------------------------
# Reinhard (2001) — match LAB mean/std
# ---------------------------------------------------------------------------


class ReinhardFitter:
    """Stain normalization via LAB-space mean/std matching."""

    def fit(self, ref_img: np.ndarray) -> "ReinhardFitter":
        lab = rgb2lab(ref_img)
        flat = lab.reshape(-1, 3)
        self.target_mean_ = flat.mean(axis=0)
        self.target_std_ = flat.std(axis=0) + 1e-8
        return self

    def transform(self, img: np.ndarray) -> np.ndarray:
        lab = rgb2lab(img)
        flat = lab.reshape(-1, 3)
        src_mean = flat.mean(axis=0)
        src_std = flat.std(axis=0) + 1e-8
        norm = (flat - src_mean) * (self.target_std_ / src_std) + self.target_mean_
        norm = np.clip(norm, [0, -127, -127], [100, 128, 128]).reshape(lab.shape)
        rgb = lab2rgb(norm)
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Macenko (2009) — SVD-based stain decomposition in OD space
# ---------------------------------------------------------------------------


class MacenkoFitter:
    """Stain normalization via Macenko's SVD-based stain decomposition."""

    def __init__(self, alpha_percentile: float = 1.0, beta: float = 0.15):
        self.alpha = alpha_percentile
        self.beta = beta

    def _stain_matrix(self, img: np.ndarray) -> np.ndarray:
        # 1. RGB to OD, drop near-white pixels.
        od = _rgb_to_od(img).reshape(-1, 3)
        od_thresh = od[(od > self.beta).any(axis=1)]
        if od_thresh.shape[0] < 10:
            od_thresh = od

        # 2. Top-2 eigenvectors of OD covariance.
        cov = np.cov(od_thresh, rowvar=False)
        _, eigvecs = np.linalg.eigh(cov)
        plane = eigvecs[:, -2:]  # ascending → take last two
        if plane[0, 0] < 0:
            plane[:, 0] = -plane[:, 0]
        if plane[0, 1] < 0:
            plane[:, 1] = -plane[:, 1]

        # 3. Project, find min/max angle.
        proj = od_thresh @ plane
        angles = np.arctan2(proj[:, 1], proj[:, 0])
        min_ang = np.percentile(angles, self.alpha)
        max_ang = np.percentile(angles, 100 - self.alpha)

        v_min = plane @ np.array([np.cos(min_ang), np.sin(min_ang)])
        v_max = plane @ np.array([np.cos(max_ang), np.sin(max_ang)])

        # 4. Hematoxylin first (higher red OD ≈ darker on red channel).
        if v_min[0] > v_max[0]:
            stain_matrix = np.stack([v_min, v_max], axis=1)
        else:
            stain_matrix = np.stack([v_max, v_min], axis=1)
        return stain_matrix  # shape (3, 2): columns are H, E stain vectors

    def _concentrations(self, od_flat: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
        # Solve OD.T = stain_matrix @ C  →  C = pinv(stain_matrix) @ OD.T
        return np.linalg.lstsq(stain_matrix, od_flat.T, rcond=None)[0]  # (2, N)

    def fit(self, ref_img: np.ndarray) -> "MacenkoFitter":
        self.target_stain_matrix_ = self._stain_matrix(ref_img)
        ref_od = _rgb_to_od(ref_img).reshape(-1, 3)
        ref_concs = self._concentrations(ref_od, self.target_stain_matrix_)
        self.target_max_C_ = np.percentile(ref_concs, 99, axis=1)
        return self

    def transform(self, img: np.ndarray) -> np.ndarray:
        h, w, _ = img.shape
        od = _rgb_to_od(img).reshape(-1, 3)
        src_stain_matrix = self._stain_matrix(img)
        src_concs = self._concentrations(od, src_stain_matrix)
        src_max_C = np.percentile(src_concs, 99, axis=1)
        scale = self.target_max_C_ / (src_max_C + 1e-8)
        norm_concs = src_concs * scale[:, None]
        norm_od = (self.target_stain_matrix_ @ norm_concs).T  # (N, 3)
        return _od_to_rgb(norm_od, (h, w, 3))


# ---------------------------------------------------------------------------
# Vahadane (2016) — sparse non-negative matrix factorization in OD space
# ---------------------------------------------------------------------------


class VahadaneFitter:
    """Stain normalization via sparse NMF on optical density."""

    def __init__(self, sparsity: float = 0.02, max_iter: int = 200, random_state: int = 42):
        self.sparsity = sparsity
        self.max_iter = max_iter
        self.random_state = random_state

    def _stain_matrix(self, img: np.ndarray) -> np.ndarray:
        od = _rgb_to_od(img).reshape(-1, 3)
        mask = od.sum(axis=1) > 0.15
        od_keep = od[mask] if mask.sum() > 100 else od

        nmf = NMF(
            n_components=2,
            init="nndsvda",
            solver="cd",
            alpha_W=self.sparsity,
            l1_ratio=1.0,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        nmf.fit(od_keep)
        components = nmf.components_  # shape (2, 3)
        # Normalise columns so each stain vector has unit norm.
        components = components / (np.linalg.norm(components, axis=1, keepdims=True) + 1e-8)
        # Hematoxylin first (higher OD on red channel = bluer).
        if components[0, 0] < components[1, 0]:
            components = components[::-1]
        return components.T  # (3, 2)

    def fit(self, ref_img: np.ndarray) -> "VahadaneFitter":
        self.target_stain_matrix_ = self._stain_matrix(ref_img)
        ref_od = _rgb_to_od(ref_img).reshape(-1, 3)
        ref_concs = np.linalg.lstsq(self.target_stain_matrix_, ref_od.T, rcond=None)[0]
        self.target_max_C_ = np.percentile(ref_concs, 99, axis=1)
        return self

    def transform(self, img: np.ndarray) -> np.ndarray:
        h, w, _ = img.shape
        od = _rgb_to_od(img).reshape(-1, 3)
        src_stain_matrix = self._stain_matrix(img)
        src_concs = np.linalg.lstsq(src_stain_matrix, od.T, rcond=None)[0]
        src_max_C = np.percentile(src_concs, 99, axis=1)
        scale = self.target_max_C_ / (src_max_C + 1e-8)
        norm_concs = src_concs * scale[:, None]
        norm_od = (self.target_stain_matrix_ @ norm_concs).T
        return _od_to_rgb(norm_od, (h, w, 3))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_fitter(method: str):
    method = method.lower()
    if method == "reinhard":
        return ReinhardFitter()
    if method == "macenko":
        return MacenkoFitter()
    if method == "vahadane":
        return VahadaneFitter()
    raise ValueError(f"Unknown stain method: {method!r}. Expected one of: reinhard, macenko, vahadane.")
