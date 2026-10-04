"""Small end-to-end GPU smoke test for ssl-data-curation."""

import numpy as np
import torch

from src import hierarchical_kmeans_gpu as hkmg
from src import hierarchical_sampling as hs
from src.clusters import HierarchicalCluster


def main() -> None:
    np.random.seed(0)
    torch.manual_seed(0)

    data = np.random.randn(320, 8).astype("float32")
    tensor = torch.tensor(data, device="cuda")

    clusters = hkmg.hierarchical_kmeans_with_resampling(
        data=tensor,
        n_clusters=[16, 4],
        n_levels=2,
        sample_sizes=[4, 2],
        n_resamples=1,
        verbose=False,
    )

    hierarchy = HierarchicalCluster.from_dict(clusters)
    selected = hs.hierarchical_sampling(hierarchy, target_size=64)

    assert len(selected) == 64
    assert selected.min() >= 0
    assert selected.max() < len(data)

    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Level-1 clusters: {len(clusters[0]['clusters'])}")
    print(f"Level-2 clusters: {len(clusters[1]['clusters'])}")
    print(f"Selected indices: {len(selected)}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
