import pickle
from typing import Callable
import numpy as np
from numpy import ndarray
import open3d.geometry as o3d_geom
from open3d.geometry import PointCloud
from utils.surface_normals import estimate_surface_normals
from torch_kmeans import CosineSimilarity, SoftKMeans
import torch 
from scipy import stats

PRINT_CLUSTERING_MESSAGES = False

class ClusterNormals:

    def __init__(self,
                 pcd: PointCloud,
                 clustering_fn: Callable[[ndarray], float],
                 normal_estimation_radius,
                 voxel_down_sample_size=None,
                 orientation_ref=np.array([0.0, 0.0, 1.0]),
                 gt_labels=None, 
                 image_id = -1
                 ):
        self._clustering_fn = clustering_fn
        self._pcd = pcd
        self._pc_downsampled, self._downsample_index_trace = estimate_surface_normals(
            self._pcd,
            voxel_down_sample_size=voxel_down_sample_size,
            normal_estimation_radius=normal_estimation_radius,
            orientation_ref=orientation_ref,
        )
        self._kd_tree = o3d_geom.KDTreeFlann(self._pc_downsampled)
        self._gt_labels = gt_labels
        self.image_id = image_id
        print(len(self._pcd.points))

    """
        Finds nearest points within radius of anchor using kdtree. Iterates through to set each downsampled point as an anchor.
        anchor: index of anchor in self._pc_downsampled
        radius: radius within which we want to look for points around the anchor
        returns: array of indices for voxels within the radius of the anchor point
    """
    def find_knn_radius(self, anchor: int, radius: float):
        num_points, idx, coordinates = self._kd_tree.search_radius_vector_3d(query=self._pc_downsampled.points[anchor],
                                                                             radius=radius)
        return idx
    

    """
    for each point cloud point we have:
        find the points with a radius using knn_radius
        find normals of those points using estimate_surface_normals
        cluster these normals for k=1,2,3
        get the cosine similarity metric for each clustering as features
    """
    def cluster_normals(self, radius, k):
        n = len(self._pc_downsampled.points) # change this to len(self.pcd)
        n=1
        # assuming k is always in order 1,2,3

        # per-cluster similarity which stores similarity for each anchor point based on the k in kmeans clustering
        pcs = np.zeros((n, len(k)))

        # neighbours per point for each anchor point
        npp = np.zeros(n)
        for a in range(n):
            r = radius

            pc_in_radius_idx = self.find_knn_radius(anchor=a, radius=r)
            model = SoftKMeans(distance=CosineSimilarity, max_iter=100, verbose=PRINT_CLUSTERING_MESSAGES)

            # Normals of the points within the radius
            bs = 1
            selected_points = np.asarray(self._pc_downsampled.normals)[pc_in_radius_idx]
            npp[a] = len(selected_points)
            if(npp[a] <= 3):
                pcs[a]= np.full(len(k), np.nan)
                continue
            
            pcs[a][0] = self.kmeans_one_cluster(selected_points)
            pc_in_radius = torch.from_numpy(np.tile((np.asarray(selected_points)), (bs, 1, 1)))
            for K in k[1:]:    
                result = model(x=pc_in_radius, k=K)
                # extract the distance of the point to its own cluster center and sum
                arr = result.inertia.cpu().numpy()[0]
                sum = np.sum(np.max(arr, axis=1))

                # convert tensors to numpy array and save
                pcs[a][K-1] = sum/len(selected_points)
        # self.save_downsampling_index_trace(file_path='./datasets/index_trace.pkl')
        # np.save(f'./datasets/cluster_similarity_{self.image_id}', pcs)
        # np.save(f'./datasets/neighbours_per_point_{self.image_id}', npp)

    
    def save_downsampling_index_trace(self, file_path: str):
        with open(file_path, 'wb') as f:
            index_trace_np_array_list = [np.asarray(int_vector) for int_vector in self._downsample_index_trace]
            pickle.dump(index_trace_np_array_list, f)
    

    """
    Performs k=1 k-means clustering on the normals passed in based on cosine similarity
    x: array of normals 
    returns: a scalar value indicating the average similarity of all the normals
    """
    def kmeans_one_cluster(self, x: np.ndarray):
        # centroid = x[np.random.randint(0, len(x))]
        centroid = np.mean(x, axis=0)
        centroids = np.full((len(x), 3), centroid)
        # take the dot product of every point with the center and add them up to get sum of similarity indices
        similarity_sum = np.sum(x * centroids)
        return similarity_sum/len(x)
    

    """
    Gets the indices for the original points in the point cloud from the downsampled voxel point which is stored
    in self._pc_downsampled.points, looks at the ground truth mask labels for these points and takes the mode of these labels
    as the true label value for the particualar voxel. It does this for each voxel and return the true label values for all voxels.
    returns: ground truth mask labels for all the points in the original pcd corresponding to each voxel 
    """
    def get_gt_labels(self):
        gt_labels_downsampled = np.zeros(len(self._pc_downsampled.points), dtype=np.int32)
        for idx, indices in enumerate(self._downsample_index_trace):
            labels = []
            for i in indices:
                labels.append(self._gt_labels[i])
            gt_labels_downsampled[idx] = stats.mode(labels)[0]
        # np.save(f'./datasets/gt_labels_downsampled_{self.image_id}', gt_labels_downsampled)
        return gt_labels_downsampled
    

    @property
    def pcd(self):
        return self._pcd
    
    @property
    def pc_downsampled(self):
        return self._pc_downsampled

