"""Run ssl-data-curation on 100 real handwritten-digit images."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

from src import hierarchical_kmeans_gpu as hkmg
from src import hierarchical_sampling as hs
from src.clusters import HierarchicalCluster


OUTPUT_DIR = Path("walkthrough_outputs")


def load_balanced_digits(per_class: int = 10):
    """Return a deterministic 100-image subset: 10 examples per digit."""
    digits = load_digits()
    chosen = np.concatenate(
        [np.flatnonzero(digits.target == label)[:per_class] for label in range(10)]
    )
    images = digits.images[chosen]
    labels = digits.target[chosen]
    # Flatten each 8x8 image to a 64-D feature vector and scale pixels to [0, 1].
    embeddings = images.reshape(len(images), -1).astype("float32") / 16.0
    return images, embeddings, labels


def label_counts(labels: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(labels, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist()))


def save_grid(images: np.ndarray, labels: np.ndarray, selected: np.ndarray) -> Path:
    cols = 8
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 1.45 * rows))
    axes = np.asarray(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for axis, index in zip(axes, selected):
        axis.imshow(images[index], cmap="gray_r")
        axis.set_title(f"idx {index} / digit {labels[index]}", fontsize=7)
        axis.axis("off")
    fig.suptitle("Hierarchically curated subset (40 of 100 images)")
    fig.tight_layout()
    output_path = OUTPUT_DIR / "curated_digits.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_statistics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    selected: np.ndarray,
    raw_clusters: list[dict],
) -> Path:
    """Create a compact, presentation-ready summary of the curation run."""
    fine_assignment = np.asarray(raw_clusters[0]["assignment"])
    fine_to_broad = np.asarray(raw_clusters[1]["assignment"])
    broad_assignment = fine_to_broad[fine_assignment]
    fine_sizes = np.bincount(fine_assignment, minlength=len(fine_to_broad))

    projection = PCA(n_components=2, random_state=7).fit_transform(embeddings)
    colors = plt.get_cmap("tab10")
    broad_colors = [colors(i) for i in range(len(np.unique(broad_assignment)))]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # A: embedding space and final selection.
    axis = axes[0, 0]
    for broad_id in np.unique(broad_assignment):
        mask = broad_assignment == broad_id
        axis.scatter(
            projection[mask, 0], projection[mask, 1],
            s=32, alpha=0.65, color=broad_colors[broad_id],
            label=f"Broad {broad_id}",
        )
    axis.scatter(
        projection[selected, 0], projection[selected, 1],
        s=85, facecolors="none", edgecolors="black", linewidths=1.2,
        label="Selected",
    )
    axis.set_title("A. PCA of 64-D image vectors")
    axis.set_xlabel("Principal component 1")
    axis.set_ylabel("Principal component 2")
    axis.legend(fontsize=8, ncol=2)

    # B: sizes of fine clusters, colored by their broad parent.
    axis = axes[0, 1]
    axis.bar(
        np.arange(len(fine_sizes)), fine_sizes,
        color=[broad_colors[parent] for parent in fine_to_broad],
    )
    axis.axhline(len(embeddings) / len(fine_sizes), color="black", ls="--", lw=1,
                 label="Mean size")
    axis.set_title("B. Fine-cluster sizes and broad parents")
    axis.set_xlabel("Fine-cluster ID")
    axis.set_ylabel("Number of images")
    axis.set_xticks(np.arange(len(fine_sizes)))
    axis.legend(fontsize=8)

    # C: label coverage before and after curation.
    axis = axes[1, 0]
    digits = np.arange(10)
    input_counts = np.bincount(labels, minlength=10)
    selected_counts = np.bincount(labels[selected], minlength=10)
    width = 0.38
    axis.bar(digits - width / 2, input_counts, width, label="Input (100)")
    axis.bar(digits + width / 2, selected_counts, width, label="Curated (40)")
    axis.set_title("C. Digit-label coverage")
    axis.set_xlabel("Digit label (not used by clustering)")
    axis.set_ylabel("Number of images")
    axis.set_xticks(digits)
    axis.legend()

    # D: semantic composition discovered in each broad cluster.
    axis = axes[1, 1]
    bottom = np.zeros(len(broad_colors), dtype=int)
    for digit in digits:
        counts = np.array([
            np.sum((broad_assignment == broad_id) & (labels == digit))
            for broad_id in range(len(broad_colors))
        ])
        axis.bar(np.arange(len(broad_colors)), counts, bottom=bottom,
                 label=str(digit))
        bottom += counts
    axis.set_title("D. Digit composition of broad clusters")
    axis.set_xlabel("Broad-cluster ID")
    axis.set_ylabel("Number of images")
    axis.set_xticks(np.arange(len(broad_colors)))
    axis.legend(title="Digit", fontsize=7, ncol=2)

    fig.suptitle(
        "Hierarchical data curation: 100 handwritten digits → 40 selected images",
        fontsize=15,
    )
    fig.tight_layout()
    output_path = OUTPUT_DIR / "presentation_statistics.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    np.random.seed(7)
    torch.manual_seed(7)
    OUTPUT_DIR.mkdir(exist_ok=True)

    images, embeddings, labels = load_balanced_digits()
    tensor = torch.tensor(embeddings, device="cuda")

    raw_clusters = hkmg.hierarchical_kmeans_with_resampling(
        data=tensor,
        n_clusters=[20, 5],
        n_levels=2,
        sample_sizes=[4, 2],
        n_resamples=1,
        sample_strategy="closest",
        verbose=False,
    )
    hierarchy = HierarchicalCluster.from_dict(raw_clusters)
    selected = hs.hierarchical_sampling(
        hierarchy,
        target_size=40,
        sampling_strategy="r",
    )

    np.save(OUTPUT_DIR / "selected_indices.npy", selected)
    grid_path = save_grid(images, labels, selected)
    statistics_path = save_statistics(embeddings, labels, selected, raw_clusters)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Input images: {len(images)}")
    print(f"Embedding matrix: {embeddings.shape}")
    print(f"Hierarchy: {len(raw_clusters[0]['clusters'])} fine -> "
          f"{len(raw_clusters[1]['clusters'])} broad clusters")
    print(f"Selected indices: {len(selected)}")
    print(f"Input label counts: {label_counts(labels)}")
    print(f"Selected label counts: {label_counts(labels[selected])}")
    print(f"Index file: {(OUTPUT_DIR / 'selected_indices.npy').resolve()}")
    print(f"Preview: {grid_path.resolve()}")
    print(f"Statistics: {statistics_path.resolve()}")


if __name__ == "__main__":
    main()
