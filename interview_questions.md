# Interview questions and spoken answers

Use the **short answer** first. Continue with the **deeper answer** only if the interviewer asks for more detail.

## 1. What problem does this paper solve?

**Short answer:**

Self-supervised learning does not require labels, but it still depends on the quality of its pre-training data. Raw web data contains many repeated common concepts and relatively few rare concepts. The paper automatically creates a more balanced subset without requiring class labels.

**Deeper answer:**

The authors propose that a good pre-training dataset should be large, diverse, and balanced. Large web pools provide scale and diversity, but their concept distribution is long-tailed. The method uses hierarchical k-means and hierarchical sampling to reduce the dominance of dense regions in an embedding space.

## 2. What is the paper's main contribution?

**Short answer:**

The main contribution is a label-free curation pipeline that combines successive k-means clustering, within-level resampling, and top-down hierarchical sampling.

**Deeper answer:**

Clustering centroids repeatedly produces a multi-level concept tree and makes the centroid distribution flatter than the original data distribution. Resampling further refines this distribution. Hierarchical sampling then balances the final sample budget across broad concepts and fine sub-concepts.

## 3. Why is vanilla k-means insufficient?

**Short answer:**

Vanilla k-means allocates more centroids to dense concepts because those concepts contribute more points and more distortion to the objective.

**Deeper answer:**

Suppose website screenshots dominate an image pool. K-means can reduce its objective by splitting those screenshots into many clusters. A rare concept may receive only one cluster. Uniform sampling from clusters would therefore still over-sample website screenshots because that concept occupies many clusters.

## 4. Why cluster centroids again?

**Short answer:**

The first-level centroids have a flatter distribution than the original data. Clustering those centroids again moves the representation closer to uniform coverage and creates broader concepts.

**Deeper answer:**

At level one, original embeddings form fine clusters. At level two, the level-one centroids become the input points and form broad clusters. Repeating this process creates a tree from original samples to fine concepts and then broader concepts.

## 5. What exactly does resampling mean here?

**Short answer:**

The method takes a limited number of representative points from each cluster, reruns k-means on that balanced proxy set, and then reassigns the full level input to the updated centroids.

**Deeper answer:**

Resampling does not create another hierarchy level. It refines the centroids within the current level. By drawing a similar number of representatives from each cluster, large clusters have less influence on the next centroid update.

## 6. Why are points closest to the centroid used during resampling?

**Short answer:**

They provide stable representatives of each cluster and reduce the influence of ambiguous boundary points or outliers.

**Critical qualification:**

Closest points can be repetitive and may reduce within-cluster diversity. That is why the resampling used to refine centroids should be distinguished from the final sampling strategy used to construct the curated dataset.

## 7. What is the difference between resampling and hierarchical sampling?

**Short answer:**

Resampling improves the cluster centroids. Hierarchical sampling uses the completed hierarchy to select the final dataset.

**Example:**

In our experiment, `sample_sizes=[4, 2]` controls centroid refinement. `target_size=40` controls the number of original images returned in the curated subset.

## 8. How does hierarchical sampling work?

**Short answer:**

It allocates the target budget from the top of the tree downward. The budget is balanced among broad clusters, then among their fine child clusters, and finally used to choose original points.

**Example:**

If 40 points are selected from five equally sized broad clusters, each receives roughly eight points. If one broad cluster has four fine children, those eight selections are then distributed among those children, subject to their available sizes.

## 9. Why does the paper prefer random final sampling?

**Short answer:**

Random sampling preserves more variation within each fine cluster.

**Deeper answer:**

The paper compares random, closest-to-centroid, and furthest-from-centroid sampling. Random performs best on most downstream benchmarks. Closest sampling is competitive but can select repetitive prototypes. Furthest sampling performs poorly because boundary points may be atypical or ambiguous.

## 10. Why use multiple hierarchy levels?

**Short answer:**

Different levels represent different semantic granularity. Low levels capture fine variations, while high levels group those variations into broad concepts.

**Evidence:**

The paper reports a large improvement from one level to two levels. Additional levels provide smaller gains. This suggests that multi-scale balancing matters, but depth has diminishing returns.

## 11. How should the number of clusters be chosen?

**Short answer:**

The paper does not provide an optimal formula. It chooses cluster counts using expected cluster sizes, hierarchy branching, compute limits, and downstream validation.

**Critical answer:**

This is a limitation. I would start from a desired average leaf size and branching factor, then measure stability, semantic coverage, computational cost, and downstream performance.

## 12. How should the number of resampling rounds be chosen?

**Short answer:**

The paper uses ten rounds by default. Zero rounds performs worse, while one hundred rounds gives only small and inconsistent additional gains.

**Critical answer:**

Ten is an empirical default rather than a theoretically optimal choice. I would stop adaptively when centroid movement or assignment changes fall below a threshold.

## 13. What role does DINOv2 play?

**Short answer:**

DINOv2 produces the image embeddings on which clustering operates. It is an upstream feature extractor and is not included in this repository.

**Critical answer:**

The encoder defines the geometry of the problem. If DINOv2 does not represent a concept meaningfully, the curation algorithm cannot balance that concept correctly. The paper's embedding ablation confirms this dependence.

## 14. Is the method genuinely unsupervised?

**Short answer:**

The curation step does not use labels, but the main image embeddings come from a model trained on manually curated ImageNet data.

**Critical answer:**

The pipeline is label-free at curation time, but it inherits supervision and selection choices from its feature extractor. The authors acknowledge this limitation.

## 15. What is the strongest evidence supporting the method?

**Short answer:**

The strongest evidence is the combination of cross-domain improvements and controlled ablations.

**Deeper answer:**

The method improves features trained on raw data across web images, text, and satellite imagery. The ablations also behave consistently: more hierarchy helps, hierarchical sampling beats flat sampling, k-means++ beats random initialization, resampling helps, and embedding quality strongly affects the result.

## 16. What does the method do well?

**Short answer:**

It scales without labels, balances more than one semantic granularity, and remains independent of a predefined class vocabulary or downstream task.

**Additional point:**

The method is conceptually simple. It builds on k-means, which makes the approach easier to distribute and operationalize than a new end-to-end neural objective.

## 17. What is the most important limitation?

**Short answer:**

The method can only balance the structure visible in its embedding space.

**Deeper answer:**

It may mistake rare noise for useful diversity. Uniform coverage is not automatically the same as high-quality coverage. A production pipeline still needs quality filtering, safety checks, deduplication, privacy controls, and outlier handling.

## 18. Can balancing ever hurt performance?

**Short answer:**

Yes. Balancing changes the task trade-off rather than universally improving every metric.

**Evidence:**

The raw image pool slightly outperforms the curated data on some dense prediction tasks. Frequent indoor and building scenes in the raw pool are relevant to those evaluations, and curation deliberately reduces their share.

## 19. Does concept balance guarantee fairness?

**Short answer:**

No. Balance in an embedding space is not equivalent to demographic fairness.

**Deeper answer:**

The paper reports somewhat narrower performance gaps on Dollar Street, but substantial income and regional gaps remain. Fairness must be measured directly and cannot be inferred from cluster balance.

## 20. How would you improve the method?

**Short answer:**

I would combine density balancing with quality and safety scores, introduce adaptive cluster counts, handle outliers explicitly, and stop resampling based on convergence.

**Further ideas:**

- Evaluate stability across random seeds.
- Combine embeddings from multiple encoders.
- Allow small validation sets to tune sampling weights for a target domain.
- Report compute and storage costs alongside downstream gains.
- Monitor cluster drift when the raw pool or encoder changes.

## 21. Explain our experiment

**Short answer:**

We ran the repository on 100 real handwritten-digit images. We represented each 8 by 8 image as a 64-dimensional raw-pixel vector, formed 20 fine clusters and five broad clusters, and selected 40 original indices.

**What it proves:**

It proves that the repository's GPU clustering, hierarchy construction, and hierarchical sampling run end to end locally.

**What it does not prove:**

It does not show that the selected subset improves self-supervised training. We did not use learned semantic embeddings, retrain a model, or evaluate a downstream task.

## 22. Why was our selected label distribution uneven?

**Short answer:**

The algorithm balanced clusters in raw-pixel space, not digit labels.

**Deeper answer:**

Although the input contained ten images of every digit, labels were hidden from clustering. Raw pixels encode stroke position and intensity imperfectly. The result demonstrates the paper's central dependency on embedding quality.

## 23. What would be the next meaningful experiment?

**Short answer:**

Use about 100 natural images, extract DINOv2 embeddings, repeat clustering across several random seeds, and evaluate whether the curated subset improves a held-out task or coverage metric.

**Stronger design:**

Compare random selection, vanilla k-means sampling, and hierarchical sampling at the same target size. Measure semantic coverage, duplicate rate, cluster stability, runtime, and downstream accuracy.

## 24. What would you say if asked whether you agree with the paper?

**Suggested answer:**

I agree with the main diagnosis that raw scale does not guarantee useful diversity and that balancing multiple semantic resolutions is valuable. The method is supported by good ablations and cross-domain results. I would be cautious about treating uniform support coverage as a universal objective because rare regions may contain noise or unsafe data, and because the embedding model defines what counts as a concept. I see the method as a strong density-balancing component inside a broader curation system, not as the entire production pipeline.

## 25. Give a one-minute summary

The paper asks how to build large, diverse, and balanced self-supervised datasets without labels. Vanilla k-means does not fully solve the problem because dense concepts receive many centroids. The authors repeatedly cluster centroids to build a hierarchy, refine each level using cluster-balanced resampling, and allocate the final sample budget top-down across the hierarchy. Their experiments show improvements over raw data across images, text, and satellite imagery, with especially strong robustness and long-tail gains. The main strength is scalable, task-agnostic balancing. The main limitation is dependence on the embedding space: the method inherits the encoder's biases and may also amplify rare noise. I would improve it with quality scoring, outlier handling, adaptive parameters, and convergence-based stopping.

## Questions to ask the interviewer

If the discussion becomes conversational, suitable questions include:

1. How does your team evaluate semantic coverage when class labels are unavailable?
2. Does your production data pipeline optimize for generic coverage or particular downstream domains?
3. How do you separate rare valuable concepts from rare noise or unsafe content?
4. Which constraint matters most at your scale: feature extraction, clustering compute, storage, or repeated reassignment?
5. Have you found cluster stability or downstream performance more useful for selecting curation hyperparameters?

## Final reminders

- Say **reassign**, not resign.
- Distinguish **resampling-clustering** from **hierarchical sampling**.
- State that level two clusters the **level-one centroids**.
- State that `selected` contains indices; the selected embedding matrix is obtained with `embeddings[selected]`.
- Do not claim that our 100-image experiment validates downstream SSL performance.
- Lead with the short answer and wait before giving the deeper explanation.
