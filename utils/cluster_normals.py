from typing import Callable
import numpy as np
from numpy import ndarray
import open3d.geometry as o3d_geom
from open3d.geometry import PointCloud
from utils.surface_normals import estimate_surface_normals
from torch_kmeans import KMeans, CosineSimilarity, SoftKMeans
import torch 



class ClusterNormals:

    def __init__(self,
                 pcd: PointCloud,
                 clustering_fn: Callable[[ndarray], float],
                 normal_estimation_radius,
                 voxel_down_sample_size=None,
                 orientation_ref=np.array([0.0, 0.0, 1.0])
                 ):
        self._clustering_fn = clustering_fn
        self._pcd = pcd

        self._pcd = estimate_surface_normals(
            self._pcd,
            voxel_down_sample_size=voxel_down_sample_size,
            normal_estimation_radius=normal_estimation_radius,
            orientation_ref=orientation_ref,
        )
        self._kd_tree = o3d_geom.KDTreeFlann(self._pcd)

    """
        finds nearest points within radius of anchor using kdtree 
        anchor: index of anchor in self.pcd
        radius: radius within which we want to look for points
    """
    def find_knn_radius(self, anchor: int, radius: float):
        num_points, idx, coordinates = self._kd_tree.search_radius_vector_3d(query=self._pcd.points[anchor],
                                                                             radius=radius)
        return idx
    

    """
    for each point cloud point we have:
        find the points with a radius using knn_radius
        find normals of those points using estimate_surface_normals
        cluster these normals
    """
    def cluster_normals(self, radius, k):
        n = 1 # change this to len(self.pcd)
        # assuming k is always in order 1,2,3
        sse = np.zeros((n, len(k)))
        
        for a in range(n):
            r = radius
            # don't use a k value of 1 since it throws an error saying k=1 is ambiguous
            # k = k
            # bs = len(k)
            pc_in_radius_idx = self.find_knn_radius(anchor=a, radius=r)
            model = SoftKMeans(distance=CosineSimilarity, max_iter=100)


            # Normals of the points within the radius
            bs = 1
            selected_points = np.asarray(self.pcd.normals)[pc_in_radius_idx]
            sse[a][0] = self.kmeans_one_cluster(selected_points)
            pc_in_radius = torch.from_numpy(np.tile((np.asarray(selected_points)), (bs, 1, 1)))
            for K in k[1:]:    
                result = model(x=pc_in_radius, k=K)
                # print(result.labels)
                # print(result.inertia.cpu().numpy().shape) 1 x len(selected_points) x k

                # extract the distance of the point to it's own cluster center and sum
                arr = result.inertia.cpu().numpy()[0]
                sum = np.sum(np.max(arr, axis=1))
                # convert tensors to numpy array and save
                sse[a][K-1] = sum

        print(sse)
    
    def kmeans_one_cluster(self, x: np.ndarray):
        centroid = np.mean(x)
        centroids = np.full((len(x), 3), centroid)
        # take the dot product of every point with the center and add them up to get sum of similarity indices
        similarity_sum = np.sum(x * centroids)
        return similarity_sum



    @property
    def pcd(self):
        return self._pcd
