"""Shared UMAP visualisations for the project's experiments.

A single UMAP fit is built on the concatenation of image and class-prompt
embeddings so both modalities live in the same 2D coordinate system. Two
plots are produced from that fit:

  - ``plot_umap_scatter`` — vanilla scatter (per-class colours + 5 prompt markers).
  - ``plot_umap_thumbnails`` — sample-image thumbnails sprinkled around the
    space using the "no two thumbnails within `min_dist`" rule, with the
    same 5 prompt markers laid on top.

The thumbnail layout follows the pattern in the reference notebook supplied
by the tutor; the prompt overlay was added so the spatial relationship
between image clusters and their class prompts is legible at a glance.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image


def fit_shared_umap(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one UMAP over (images || prompts) and return the two views.

    Returns ``(image_2d, text_2d)`` so images and prompts can be plotted in
    the same coordinate system.
    """
    image_embeddings = np.asarray(image_embeddings)
    text_embeddings = np.asarray(text_embeddings)
    combined = np.concatenate([image_embeddings, text_embeddings], axis=0)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    combined_2d = reducer.fit_transform(combined)
    n_img = len(image_embeddings)
    return combined_2d[:n_img], combined_2d[n_img:]


def _safe_open(path, size: tuple[int, int]) -> np.ndarray:
    try:
        im = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
        return np.asarray(im)
    except Exception:
        return np.full((size[1], size[0], 3), 200, dtype=np.uint8)


def class_color_palette(n_classes: int):
    """Return the per-class colour vector used across the project's UMAPs.

    Using a fixed palette (tab10) keeps the colours consistent between
    the scatter dots, the thumbnail borders, and any per-class
    annotations in the same figure.
    """
    return plt.cm.tab10(np.arange(n_classes))


def _draw_class_scatter(ax, coords, labels, class_names, *, alpha, s, colors=None):
    coords = np.asarray(coords)
    labels = np.asarray(labels)
    if colors is None:
        colors = class_color_palette(len(class_names))
    for i, name in enumerate(class_names):
        mask = labels == i
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=s, alpha=alpha, label=name, color=colors[i],
        )


def _draw_prompt_markers(ax, text_2d, class_names):
    for (x, y), name in zip(text_2d, class_names):
        ax.scatter(
            x, y, s=320, marker="*",
            edgecolor="black", linewidths=1.2, color="white", zorder=5,
        )
        ax.annotate(
            name, (x, y),
            xytext=(10, 8), textcoords="offset points",
            fontsize=10, fontweight="bold", zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white", edgecolor="black", alpha=0.85,
            ),
        )


def plot_umap_scatter(
    image_2d: np.ndarray,
    image_labels: Sequence[int],
    text_2d: np.ndarray | None = None,
    class_names: Sequence[str] | None = None,
    *,
    title: str = "Shared UMAP — image embeddings + class prompts",
):
    """Per-class scatter, optionally with class-prompt markers overlaid.

    ``text_2d=None`` skips the prompt markers — appropriate for ablations
    that have no text encoder.
    """
    if class_names is None:
        labels_arr = np.asarray(image_labels)
        n_classes = int(labels_arr.max() + 1)
        class_names = [f"class {i}" for i in range(n_classes)]
    colors = class_color_palette(len(class_names))
    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_class_scatter(ax, image_2d, image_labels, class_names, alpha=0.55, s=10, colors=colors)
    if text_2d is not None:
        _draw_prompt_markers(ax, text_2d, class_names)
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    return fig


def plot_umap_thumbnails(
    image_2d: np.ndarray,
    image_paths: Sequence,
    image_labels: Sequence[int],
    text_2d: np.ndarray | None = None,
    class_names: Sequence[str] | None = None,
    *,
    max_thumbs: int = 30,
    thumb_size: tuple[int, int] = (56, 56),
    min_dist_frac: float = 0.05,
    seed: int = 42,
    border_width: float = 2.5,
    title: str = "Shared UMAP — sample images + class prompts",
):
    """Tutor-style thumbnail overlay with class-coloured borders.

    If ``text_2d`` and ``class_names`` are both provided, the five class-prompt
    embeddings are drawn as star markers on top of the thumbnails (the
    standard shared-UMAP variant). For ablations with no text encoder
    (e.g. the plain image classifier), pass ``text_2d=None`` to skip the
    prompt overlay.

    Each thumbnail's border is coloured to match its true class, using the
    same palette as the scatter dots underneath — this makes class-mismatch
    cases (a thumbnail of class A sitting in class B's cluster) visually
    obvious.

    ``min_dist_frac`` is expressed as a fraction of the smaller axis range
    so the layout scales automatically with whatever UMAP produces.
    """
    coords = np.asarray(image_2d)
    paths = list(image_paths)
    labels = np.asarray(image_labels)
    rng = np.random.default_rng(seed)

    n_classes = len(class_names) if class_names is not None else int(labels.max() + 1)
    if class_names is None:
        class_names = [f"class {i}" for i in range(n_classes)]
    colors = class_color_palette(n_classes)

    fig, ax = plt.subplots(figsize=(11, 8))
    _draw_class_scatter(ax, coords, labels, class_names, alpha=0.30, s=6, colors=colors)

    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    min_dist = min_dist_frac * float(min(x_range, y_range))

    idx = np.arange(len(coords))
    rng.shuffle(idx)
    placed: list[tuple[float, float]] = []
    for i in idx:
        x, y = float(coords[i, 0]), float(coords[i, 1])
        if any((x - px) ** 2 + (y - py) ** 2 < min_dist ** 2 for (px, py) in placed):
            continue
        thumb = _safe_open(paths[i], size=thumb_size)
        cls = int(labels[i])
        ab = AnnotationBbox(
            OffsetImage(thumb, zoom=1.0),
            (x, y),
            frameon=True,
            pad=0.25,
            bboxprops=dict(linewidth=border_width, edgecolor=colors[cls]),
        )
        ax.add_artist(ab)
        placed.append((x, y))
        if len(placed) >= max_thumbs:
            break

    if text_2d is not None:
        _draw_prompt_markers(ax, text_2d, class_names)

    ax.set_title(f"{title} (n={len(placed)})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    return fig
