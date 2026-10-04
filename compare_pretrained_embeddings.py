"""Compare raw pixels with ImageNet-pretrained ResNet-18 embeddings."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from torchvision.models import ResNet18_Weights, resnet18

from real_data_walkthrough import load_balanced_digits
from src import hierarchical_kmeans_gpu as hkmg
from src import hierarchical_sampling as hs
from src.clusters import HierarchicalCluster


OUTPUT_DIR = Path("walkthrough_outputs")
COLORS = {"Raw pixels": "#E76F51", "Pretrained ResNet-18": "#2D6CDF"}


def extract_resnet_embeddings(images: np.ndarray) -> np.ndarray:
    """Extract 512-D global features from an ImageNet-pretrained ResNet-18."""
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().cuda()

    tensor = torch.tensor(images, dtype=torch.float32, device="cuda")[:, None] / 16.0
    tensor = tensor.repeat(1, 3, 1, 1)
    tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
    tensor = (tensor - mean) / std

    with torch.inference_mode():
        embeddings = model(tensor)
        embeddings = F.normalize(embeddings, dim=1)
    return embeddings.cpu().numpy().astype("float32")


def run_curation(embeddings: np.ndarray, seed: int = 7):
    np.random.seed(seed)
    torch.manual_seed(seed)
    raw_clusters = hkmg.hierarchical_kmeans_with_resampling(
        data=torch.tensor(embeddings, device="cuda"),
        n_clusters=[20, 5],
        n_levels=2,
        sample_sizes=[4, 2],
        n_resamples=1,
        sample_strategy="closest",
        verbose=False,
    )
    hierarchy = HierarchicalCluster.from_dict(raw_clusters)
    selected = hs.hierarchical_sampling(
        hierarchy, target_size=40, sampling_strategy="r"
    )
    return raw_clusters, selected


def metrics(labels: np.ndarray, raw_clusters: list[dict], selected: np.ndarray):
    counts = np.bincount(labels[selected], minlength=10)
    probabilities = counts[counts > 0] / counts.sum()
    entropy = -(probabilities * np.log(probabilities)).sum() / np.log(10)
    balance = 1.0 - np.abs(counts - 4).sum() / 80.0

    fine_assignment = np.asarray(raw_clusters[0]["assignment"])
    purity_total = 0
    for cluster_id in range(20):
        members = labels[fine_assignment == cluster_id]
        if len(members):
            purity_total += np.bincount(members, minlength=10).max()
    purity = purity_total / len(labels)
    nmi = normalized_mutual_info_score(labels, fine_assignment)
    return {
        "counts": counts,
        "coverage": int(np.count_nonzero(counts)),
        "entropy": float(entropy),
        "balance": float(balance),
        "purity": float(purity),
        "nmi": float(nmi),
    }


def save_comparison(
    labels: np.ndarray,
    embeddings_by_name: dict[str, np.ndarray],
    selected_by_name: dict[str, np.ndarray],
    metrics_by_name: dict[str, dict],
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for axis, (name, embeddings) in zip(axes[0], embeddings_by_name.items()):
        projection = PCA(n_components=2, random_state=7).fit_transform(embeddings)
        selected = selected_by_name[name]
        scatter = axis.scatter(
            projection[:, 0], projection[:, 1], c=labels, cmap="tab10", s=38, alpha=0.65
        )
        axis.scatter(
            projection[selected, 0], projection[selected, 1], s=95,
            facecolors="none", edgecolors="black", linewidths=1.3,
        )
        axis.set_title(f"{name}: PCA (black rings = selected)")
        axis.set_xlabel("Principal component 1")
        axis.set_ylabel("Principal component 2")
    legend = axes[0, 1].legend(*scatter.legend_elements(), title="Digit", ncol=2)
    axes[0, 1].add_artist(legend)

    axis = axes[1, 0]
    digits = np.arange(10)
    width = 0.36
    for offset, (name, result) in zip([-width / 2, width / 2], metrics_by_name.items()):
        axis.bar(digits + offset, result["counts"], width, label=name, color=COLORS[name])
    axis.axhline(4, color="black", ls="--", lw=1, label="Ideal count")
    axis.set_title("Selected label distribution (40 images)")
    axis.set_xlabel("Digit label, used only for evaluation")
    axis.set_ylabel("Selected images")
    axis.set_xticks(digits)
    axis.legend(fontsize=9)

    axis = axes[1, 1]
    metric_names = ["Coverage", "Entropy", "Balance", "Purity", "NMI"]
    x = np.arange(len(metric_names))
    for offset, (name, result) in zip([-width / 2, width / 2], metrics_by_name.items()):
        values = [result["coverage"] / 10, result["entropy"], result["balance"],
                  result["purity"], result["nmi"]]
        axis.bar(x + offset, values, width, label=name, color=COLORS[name])
    axis.set_title("Comparison metrics (higher is better)")
    axis.set_ylim(0, 1.08)
    axis.set_xticks(x, metric_names)
    axis.set_ylabel("Normalized score")
    axis.legend(fontsize=9)

    fig.suptitle("Hierarchical curation: raw pixels vs pretrained neural embeddings", fontsize=16)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "embedding_comparison.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    images, raw_pixels, labels = load_balanced_digits()
    raw_pixels = F.normalize(torch.tensor(raw_pixels), dim=1).numpy()
    resnet_embeddings = extract_resnet_embeddings(images)

    embeddings_by_name = {
        "Raw pixels": raw_pixels,
        "Pretrained ResNet-18": resnet_embeddings,
    }
    selected_by_name = {}
    metrics_by_name = {}
    for name, embeddings in embeddings_by_name.items():
        clusters, selected = run_curation(embeddings)
        selected_by_name[name] = selected
        metrics_by_name[name] = metrics(labels, clusters, selected)
        np.save(OUTPUT_DIR / f"selected_{name.lower().replace(' ', '_').replace('-', '_')}.npy", selected)

    output_path = save_comparison(
        labels, embeddings_by_name, selected_by_name, metrics_by_name
    )
    print(f"Raw embedding shape: {raw_pixels.shape}")
    print(f"ResNet embedding shape: {resnet_embeddings.shape}")
    for name, result in metrics_by_name.items():
        print(f"{name}: {result}")
    print(f"Comparison plot: {output_path.resolve()}")


if __name__ == "__main__":
    main()
