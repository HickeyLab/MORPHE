from scipy.spatial.distance import cdist
from skimage.feature import canny
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
# 1. Clustering Evaluation for Original and Generated Maps
from sklearn.metrics import adjusted_rand_score
import numpy as np

def cluster_cell_types(true_map, generated_map, num_clusters=25, mode='full'):
    print(1)
    """
    Compares clustering structure between true and generated maps.

    Parameters:
        true_map: [H, W] ndarray
        generated_map: [H, W] ndarray
        mode: 'overlap' (only where both have values >0), or 'full' (flattened)

    Returns:
        dict: {'ARI': ..., 'NMI': ...}
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


    true_map = true_map.squeeze()
    generated_map = generated_map.squeeze()

    if mode == 'overlap':
        mask = (true_map > 0) & (generated_map > 0)
        true_labels = true_map[mask].flatten()
        generated_labels = generated_map[mask].flatten()
    elif mode == 'full':
        assert true_map.shape == generated_map.shape
        true_labels = true_map.flatten()
        generated_labels = generated_map.flatten()
    else:
        raise ValueError("mode must be 'overlap' or 'full'")
    true_labels = true_labels.cpu().numpy()
    generated_labels = generated_labels.cpu().numpy()

    if len(true_labels) == 0 or len(generated_labels) == 0:
        return {'ARI': 0.0, 'NMI': 0.0}

    ari = adjusted_rand_score(true_labels, generated_labels)
    nmi = normalized_mutual_info_score(true_labels, generated_labels)

    return {'ARI': ari, 'NMI': nmi}



# 2. Neighborhood Similarity Evaluation
from sklearn.neighbors import NearestNeighbors

def neighborhood_similarity(true_map, generated_map, k=20, max_per_type=50):
        print(2)
        """
        Compare neighborhood type distributions for each cell type separately.
        The number of classes is inferred automatically.
        For speed, each cell type is limited to at most `max_per_type` cells.

        Parameters:
        - true_map: [1,H,W] or [H,W] tensor or ndarray
        - generated_map: same shape as above
        - k: number of neighbors for each cell
        - max_per_type: maximum number of cells to sample per cell type

        Returns:
        - per_type_scores: dict {cell_type: cosine_similarity}
        - weighted_avg_score: float, weighted by number of sampled cells per type
        """

        # Convert tensors to numpy arrays if needed
        true_map = true_map.squeeze().cpu().numpy() if hasattr(true_map, 'cpu') else true_map
        generated_map = generated_map.squeeze().cpu().numpy() if hasattr(generated_map, 'cpu') else generated_map

        # Infer number of classes
        all_labels = np.union1d(np.unique(true_map), np.unique(generated_map))
        print(all_labels)
        num_classes = len(all_labels)
        print("Number of classes: {}".format(num_classes))

        def get_neighbor_type_distribution(cell_map, target_type):
            #print(target_type)
            coords = np.argwhere(cell_map == target_type)
            types = cell_map

            if len(coords) < 2:
                return None, 0  # not enough cells

            # Sample if too many cells of this type
            if len(coords) > max_per_type:
                idx = np.random.choice(len(coords), max_per_type, replace=False)
                coords = coords[idx]

            knn = NearestNeighbors(n_neighbors=min(k + 1, len(coords)), metric='euclidean').fit(coords)
            _, indices = knn.kneighbors(coords)

            neighbor_coords = coords[indices[:, 1:]]  # exclude self
            neighbor_types = types[neighbor_coords[..., 0], neighbor_coords[..., 1]]  # [N, k]

            N = neighbor_types.shape[0]
            hist = np.zeros((N, num_classes), dtype=np.float32)
            for i in range(N):
                for t in neighbor_types[i]:
                    hist[i, t] += 1
            hist /= hist.sum(axis=1, keepdims=True)
            return hist, len(coords)

        per_type_scores = {}
        weighted_sum = 0.0
        total_weight = 0

        for t in range(1, num_classes):  # skip background (0)
            hist_true, count_true = get_neighbor_type_distribution(true_map, t)
            hist_gen, count_gen = get_neighbor_type_distribution(generated_map, t)

            if hist_true is None or hist_gen is None:
                continue

            avg_true = hist_true.mean(axis=0) + 1e-8
            avg_gen = hist_gen.mean(axis=0) + 1e-8

            cosine_sim = np.dot(avg_true, avg_gen) / (np.linalg.norm(avg_true) * np.linalg.norm(avg_gen))
            per_type_scores[t] = cosine_sim

            # Weighted average by number of matched samples
            weight = min(count_true, count_gen)
            weighted_sum += weight * cosine_sim
            total_weight += weight

        weighted_avg_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        return weighted_avg_score


# 3. Cell Type Distribution Evaluation
def cell_type_distribution(true_map, generated_map):
    print(3)
    """
    Compare the global cell type distribution between the true and generated maps
    using KL divergence. The number of classes is automatically determined from both maps.

    Parameters:
    - true_map (torch.Tensor): Ground truth cell map of shape [1, H, W] or [H, W]
    - generated_map (torch.Tensor): Generated cell map of shape [1, H, W] or [H, W]

    Returns:
    - similarity_score (float): Inverse of KL divergence, bounded in (0, 1]
    """

    # Convert tensors to numpy arrays
    true_map = true_map.squeeze().cpu().numpy()
    generated_map = generated_map.squeeze().cpu().numpy()

    # Automatically determine the number of classes from both maps
    unique_true = np.unique(true_map)
    unique_gen = np.unique(generated_map)
    all_labels = np.union1d(unique_true, unique_gen)
    num_classes = int(all_labels.max()) + 1  # labels assumed to be non-negative integers

    # Compute histograms for each map using full label range
    true_hist, _ = np.histogram(true_map, bins=np.arange(num_classes + 1))
    gen_hist, _ = np.histogram(generated_map, bins=np.arange(num_classes + 1))

    # If either map is empty, return score of 0
    if true_hist.sum() == 0 or gen_hist.sum() == 0:
        return 0.0

    # Smooth and normalize
    eps = 1e-8
    true_hist = (true_hist + eps) / (true_hist.sum() + eps * num_classes)
    gen_hist = (gen_hist + eps) / (gen_hist.sum() + eps * num_classes)

    # Compute KL divergence
    kl_div = np.sum(true_hist * np.log(true_hist / gen_hist))
    print("KL divergence =", kl_div)

    # Similarity score: larger = more similar
    similarity_score = 1 / (1 + kl_div)
    return similarity_score




# 4. Boundary Continuity Evaluation



# 5. Spatial Distribution Fidelity
def spatial_distribution_fidelity(true_map, generated_map):
    print(5)
    """
    Evaluate the spatial distribution fidelity of cells in the generated map by comparing distances of cell centers.

    Parameters:
    - true_map: The ground truth cell map (shape [512, 512]).
    - generated_map: The generated cell map (shape [512, 512]).

    Returns:
    - fidelity_score: The spatial distribution fidelity score (higher is better).
    """
    true_map = true_map.squeeze().cpu().numpy()
    generated_map = generated_map.squeeze().cpu().numpy()
    true_cells = np.argwhere(true_map > 0)
    generated_cells = np.argwhere(generated_map > 0)

    # If no cells are present in either map, return zero fidelity
    if len(true_cells) == 0 or len(generated_cells) == 0:
        return 0.0

    true_cell_centers = np.mean(true_cells, axis=0)
    generated_cell_centers = np.mean(generated_cells, axis=0)

    # Calculate the Euclidean distance between the centers of the two cell distributions
    distance = np.linalg.norm(true_cell_centers - generated_cell_centers)

    # Lower distance means better fidelity
    return 1 / (1 + distance)


# Comprehensive Score Calculation
def calculate_comprehensive_score(cell_type_clustering, neighborhood_similarity,
                                  cell_type_distribution,
                                  spatial_distribution_fidelity):
    """
    Calculate the comprehensive score by combining different evaluation metrics.

    Parameters:
    - cell_type_clustering: Cell type clustering accuracy (standardized value in the range [0, 1])
    - neighborhood_similarity: Cell neighborhood similarity (standardized value in the range [0, 1])
    - cell_type_distribution: Cell type distribution (standardized value in the range [0, 1])
    - boundary_continuity: Boundary continuity (standardized value in the range [0, 1])
    - spatial_distribution_fidelity: Spatial distribution fidelity (standardized value in the range [0, 1])

    Returns:
    - comprehensive_score: The final comprehensive score combining all evaluation metrics.
    """

    # Set the weights for each metric
    weights = {
        "cell_type_clustering": 0.3,
        "neighborhood_similarity": 0.25,
        "cell_type_distribution": 0.2,
        "boundary_continuity": 0.15,
        "spatial_distribution_fidelity": 0.1
    }

    # Compute the comprehensive score as a weighted sum of all the metrics
    comprehensive_score = (
            weights["cell_type_clustering"] * cell_type_clustering +
            weights["neighborhood_similarity"] * neighborhood_similarity +
            weights["cell_type_distribution"] * cell_type_distribution +
            weights["spatial_distribution_fidelity"] * spatial_distribution_fidelity
    )

    return comprehensive_score
