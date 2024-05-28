import pickle
import numpy as np
from numpy import ndarray, array
import open3d.geometry as o3d_geom
from open3d.geometry import PointCloud
from open3d.utility import IntVector
from src.utils.surface_normals import estimate_surface_normals
from torch_kmeans import CosineSimilarity, SoftKMeans
import torch 
from scipy import stats

PRINT_CLUSTERING_MESSAGES = False

class ClusterNormals:

    def __init__(self,
                 pcd: PointCloud,
                 normal_estimation_radius: float,
                 voxel_down_sample_size: float = None,
                 orientation_ref: ndarray = array([0.0, 0.0, 1.0]),
                 gt_labels=None, 
                 image_id=-1
                 ):
        self._pcd = pcd
        self._pcd_downsampled, self._downsample_index_trace = estimate_surface_normals(
            self._pcd,
            voxel_down_sample_size=voxel_down_sample_size,
            normal_estimation_radius=normal_estimation_radius,
            orientation_ref=orientation_ref,
        )
        self._kd_tree = o3d_geom.KDTreeFlann(self._pcd_downsampled)
        self._gt_labels = gt_labels
        self.image_id = image_id

    def find_points_in_radius(self, anchor: int, radius: float) -> IntVector:
        """
        Finds all the points within a radius of an anchor point using a kd-tree.
        Iterates through to set each downsampled point as an anchor.

        Arguments:
            anchor: Index of anchor point in `self._pcd_downsampled`.
            radius: Radius within which we want to look for points around the anchor.

        Returns:
            Array of indices for voxels within the radius of the anchor point.
        """
        num_points, indices, coordinates = self._kd_tree.search_radius_vector_3d(
            query=self._pcd_downsampled.points[anchor], radius=radius
        )
        return indices

    def cluster_normals(self, radius, k):
        """
        For each point in a point cloud do the following:
            1. Find the neighbouring points within a radius of the point.
            2. Estimate the normals of the neighbouring points.
            3. Cluster the estimated normals using k-means with k=1,2,3.
            4. Calculate the mean intra-cluster cosine similarity distance for each cluster.
        """
        n = len(self._pcd_downsampled.points)
        print(f"voxels:{n}")

        # Assume k is always in order 1, 2, 3
        # Per-cluster similarity which stores similarity for each anchor point based on the k in kmeans clustering
        pcs = np.zeros((n, len(k)))

        # neighbours per point for each anchor point
        npp = np.zeros(n)
        for a in range(n):
            r = radius

            pc_in_radius_idx = self.find_points_in_radius(anchor=a, radius=r)
            model = SoftKMeans(distance=CosineSimilarity, max_iter=100, verbose=PRINT_CLUSTERING_MESSAGES)

            # Normals of the points within the radius
            bs = 1
            selected_points = np.asarray(self._pcd_downsampled.normals)[pc_in_radius_idx]
            npp[a] = len(selected_points)
            if npp[a] <= 3:
                pcs[a] = np.full(len(k), np.nan)
                continue
            
            pcs[a][0] = k_means_one_cluster(selected_points)
            pc_in_radius = torch.from_numpy(np.tile((np.asarray(selected_points)), (bs, 1, 1)))
            for K in k[1:]:    
                result = model(x=pc_in_radius, k=K)
                # extract the distance of the point to its own cluster center and sum
                arr = result.inertia.cpu().numpy()[0]
                sm = np.sum(np.max(arr, axis=1))

                # convert tensors to numpy array and save
                pcs[a][K-1] = sm/len(selected_points)
        # self.save_downsampling_index_trace(file_path='./datasets/index_trace.pkl')
        print(f"neighbors:{len(npp)}")
        np.save(f'./datasets/cluster_similarity_{self.image_id}', pcs)
        np.save(f'./datasets/neighbours_per_point_{self.image_id}', npp)

    def save_downsampling_index_trace(self, file_path: str):
        with open(file_path, 'wb') as f:
            index_trace_np_array_list = [np.asarray(int_vector) for int_vector in self._downsample_index_trace]
            pickle.dump(index_trace_np_array_list, f)

    def get_downsampled_gt_labels(self) -> ndarray:
        """
        Finds the ground-truth mask labels for all points in the downsampled point cloud. Does this by tracing the index
        of each downsampled point to get its original point cloud point(s) and taking the mode of their ground-truth
        labels.

        Returns:
            Ground-truth mask labels for all the points in the downsampled point cloud.
        """
        gt_labels_downsampled = np.zeros(len(self._pcd_downsampled.points), dtype=np.int32)
        for downsampled_index, orig_indices in enumerate(self._downsample_index_trace):
            labels = []
            for orig_index in orig_indices:
                labels.append(self._gt_labels[orig_index])
            mode = stats.mode(labels)[0]
            gt_labels_downsampled[downsampled_index] = mode
        np.save(f'./datasets/gt_labels_downsampled_{self.image_id}', gt_labels_downsampled)
        return gt_labels_downsampled

    @property
    def pcd(self):
        return self._pcd
    
    @property
    def pcd_downsampled(self):
        return self._pcd_downsampled


def k_means_one_cluster(x: ndarray) -> float:
    """
    Performs k-means clustering with k=1 on some examples `x` based on cosine similarity.

    Arguments:
        x: a matrix of examples.

    Returns:
        A scalar value indicating the mean cosine similarity distance between all normals.
    """
    # centroid = x[np.random.randint(0, len(x))]
    centroid = np.mean(x, axis=0)
    centroids = np.full((len(x), 3), centroid)
    # take the dot product of every point with the center and add them up to get sum of similarity indices
    similarity_sum = np.sum(x * centroids)
    return similarity_sum/len(x)
