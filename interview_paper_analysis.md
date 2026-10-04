# Interview preparation: Automatic Data Curation for Self-Supervised Learning

Paper: Huy V. Vo et al., *Automatic Data Curation for Self-Supervised Learning: A Clustering-Based Approach* (2024).

## A concise reading of the paper

The paper starts from a practical observation: self-supervised learning removes the need for labels, but it does not remove the need for good training data. Large web collections are diverse, yet their concepts follow a long-tailed distribution. Repeated concepts can dominate training while rare concepts receive too little capacity.

The authors propose three desired properties for a pre-training dataset:

- **Large:** enough examples to support high-capacity models.
- **Diverse:** broad coverage of the underlying data support.
- **Balanced:** less concentration on dominant concepts.

Because concepts and class labels are unavailable, the method works in an embedding space. For images, the paper uses DINOv2 features; for text, it uses SBERT features. The repository begins after this feature-extraction stage.

## The technical argument

### Why vanilla k-means is insufficient

Vanilla k-means minimizes total within-cluster distortion. Dense regions contribute more points and therefore more distortion. K-means reduces that distortion by allocating many centroids to dominant concepts. Uniformly sampling from the resulting clusters can still over-sample those concepts because they occupy many clusters.

The paper argues that k-means centroids have a flatter distribution than the input data, although in high dimensions the flattening from one k-means application is weak. Successively clustering centroids moves the representation closer to a uniform distribution over the support of the embedded data.

### Hierarchical k-means

At level 1, k-means clusters the original embeddings. At level 2, k-means clusters the level-1 centroids. Additional levels repeat this operation. The result is a tree in which leaves are original data points, low-level nodes represent fine concepts, and high-level nodes represent broader concepts.

### Resampling-clustering

At each level, the method selects a small number of points close to each centroid, reruns k-means on this more balanced set, and reassigns the complete level input to the updated centroids. Repeating this step further flattens centroid allocation without reducing the number of inputs passed to the next iteration.

The ablation is important: no resampling performs worse than the default ten rounds, while one hundred rounds produces only small additional gains. This suggests diminishing returns and supports ten as a practical default rather than a theoretically optimal value.

### Hierarchical sampling

After constructing the tree, a target sample budget is allocated top-down. The method first balances the budget among broad clusters, then among their child clusters, and finally samples original data points from the leaf clusters. This preserves balance at more than one semantic granularity.

Random leaf sampling performs best in the reported ablation. Closest-to-centroid sampling follows closely, while furthest-point sampling performs poorly, probably because boundary points can be atypical or ambiguous.

## What works well

### 1. The method addresses a real mismatch between scale and quality

The paper does not assume that more raw data automatically improves self-supervised learning. It identifies concept imbalance as a mechanism that can make scale ineffective.

### 2. The approach is label-free and task-agnostic

The method does not require a manually defined class vocabulary or a downstream task. This distinguishes it from retrieval pipelines anchored to ImageNet or another curated seed dataset.

### 3. The hierarchy represents multiple semantic resolutions

A single clustering can balance only one granularity. The hierarchy can represent broad concepts and fine variations simultaneously. The ablation shows a large improvement from one to two levels, followed by smaller gains from additional levels.

### 4. The paper tests the method beyond natural images

The authors apply the approach to web images, text, and satellite imagery. This supports the claim that the algorithm operates on embeddings rather than image-specific assumptions.

### 5. The ablations examine consequential design choices

The paper compares hierarchy depth, flat versus hierarchical sampling, random versus centroid-based leaf sampling, k-means++ versus random initialization, cluster counts, resampling rounds, and base embeddings. These experiments make the engineering choices easier to evaluate.

## Limitations and critical questions

### 1. Uniform coverage of embedding support is not always the desired objective

The method assumes that flattening density over the embedding support improves pre-training. Rare regions can instead contain noise, corrupted data, unsafe content, or artifacts. Equalizing support can amplify these regions. A production system needs quality, safety, privacy, and deduplication controls in addition to density balancing.

### 2. The embedding model defines the concepts

The method is only as good as the geometry of its feature space. The paper's ablation confirms that changing the base embeddings substantially changes downstream quality. ImageNet-trained DINOv2 features also introduce manual curation indirectly, so the pipeline is not fully independent of curated supervision.

### 3. Hyperparameter selection is heuristic

The theory does not specify the optimal number of levels, clusters per level, samples per cluster, or resampling rounds. The paper chooses these values through intuition, convenience, and empirical validation. This creates a large tuning and compute burden at a new scale or in a new domain.

### 4. Better balance can hurt tasks aligned with dense concepts

The curated image data improves many robustness and long-tail benchmarks, but the raw pool slightly outperforms it on some dense prediction tasks. The raw pool contains many indoor and building scenes relevant to those benchmarks; balancing intentionally reduces their share. Curation therefore changes the task trade-off rather than universally improving every downstream metric.

### 5. The small fairness gains do not resolve representation bias

The paper reports narrower performance gaps on Dollar Street, but substantial income and regional gaps remain. Concept balance in one embedding space is not equivalent to demographic fairness.

### 6. Scalability comes with operational complexity

The large image experiment clusters hundreds of millions of embeddings with millions of clusters. Distributed k-means++, checkpointing, storage, and repeated reassignment remain expensive even if they are cheaper than manual annotation.

### 7. Cluster interpretability is mostly qualitative

The hierarchy is visually coherent in selected examples, but a systematic measure of semantic purity, stability across random seeds, and sensitivity to embedding drift would make the argument stronger.

## Improvements I would investigate

1. **Combine density balance with quality scores.** Allocate sampling weight using both cluster rarity and an independent quality or safety model.
2. **Use adaptive cluster counts.** Stop splitting when a cluster is compact or too small, rather than fixing every level globally.
3. **Stop resampling by convergence.** Measure assignment changes or centroid movement and stop when further rounds add little value.
4. **Test stability.** Repeat clustering with different seeds and report overlap, cluster consistency, and downstream variance.
5. **Use multiple embedding views.** Combine features from different encoders to reduce dependence on the biases of one representation.
6. **Add explicit outlier handling.** Separate rare coherent concepts from isolated noise before applying balanced sampling.
7. **Support downstream constraints when appropriate.** Keep the generic hierarchy, but allow a small validation set to tune sampling weights for a target domain.
8. **Evaluate efficiency as a first-class result.** Report compute, memory, storage, wall-clock time, and quality gained per unit of curation cost.

## Our 100-image walkthrough

Our demonstration uses 100 handwritten-digit images, with ten images per known digit. We flatten each 8 by 8 image into a 64-dimensional vector and run:

```python
n_clusters=[20, 5]
sample_sizes=[4, 2]
n_resamples=1
target_size=40
```

This produces 20 fine clusters, five broad clusters, and a final subset of 40 indices. The experiment validates the repository's complete clustering and sampling path on the GPU.

The selected labels are imbalanced even though the input labels are balanced. This is a useful result rather than a failure of execution. The algorithm balances clusters in the supplied feature geometry, not ground-truth labels. Flattened pixels provide a weak semantic representation, so the experiment demonstrates why the choice of encoder matters. A stronger follow-up would extract DINOv2 embeddings from natural images and repeat the same analysis.

## Likely interview questions

### Why does vanilla k-means preserve some long-tail bias?

Its objective weights every data point. Dense concepts create more distortion and therefore receive more centroids. Sampling evenly from centroids can still sample a dominant concept many times.

### Why cluster centroids again?

Centroids have a flatter density than the original data. Clustering them repeatedly moves the highest-level centroids closer to uniform coverage of the embedded support and creates a multi-resolution concept tree.

### Why resample within a level?

Hierarchical clustering alone reduces the input size at every level. Resampling creates a balanced proxy input, updates the same number of centroids, and then reassigns the full level input. The operation can therefore repeat without collapsing the hierarchy further.

### Why did random final sampling beat closest-to-centroid sampling?

Random sampling retains more within-cluster variation. Closest points are representative but can be visually repetitive. Furthest points emphasize boundaries and possible outliers.

### Does the method discover true classes?

No. It discovers structure induced by the encoder and distance metric. A cluster can represent an object, style, viewpoint, background, or another correlated feature.

### How would you choose the number of clusters?

I would start from the desired average leaf size and target hierarchy branching factor, then validate cluster stability, coverage, compute cost, and downstream performance. The paper provides intuition but no optimal selection rule.

### What is the strongest evidence in the paper?

The cross-domain improvements and the ablations are stronger than any single headline benchmark. The method improves raw data in images, text, and satellite imagery, and the reported gains respond predictably to hierarchy depth, initialization, resampling, and embedding quality.

### What is the most important limitation?

The encoder defines the semantic geometry. If the representation ignores a rare but important concept, hierarchical balancing cannot recover it. If it separates noise into its own region, the method may over-sample that noise.

### How should we interpret our demonstration?

It is an implementation and reasoning demonstration, not evidence that the small curated subset improves a newly trained SSL model. To test that claim, we would need learned embeddings, a downstream training stage, repeated seeds, and a held-out evaluation.

## A strong closing position

The paper offers a principled and scalable alternative to label-based curation. Its main contribution is not k-means alone, but the combination of successive centroid clustering, within-level resampling, and top-down sampling. The approach works well when the embedding space is semantically meaningful and the goal is broad coverage. Its central risk is the same dependency: the method inherits the encoder's biases and can mistake rare noise for valuable diversity. A stronger production pipeline would combine hierarchical density balancing with quality, safety, stability, and task-aware validation.
