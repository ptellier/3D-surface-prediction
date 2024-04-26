from typing import Callable
import numpy as np
from numpy import ndarray
import open3d.geometry as o3d_geom
from open3d.geometry import PointCloud
from utils.surface_normals import estimate_surface_normals
from torch_kmeans import KMeans, CosineSimilarity
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
        self._kd_tree = o3d_geom.KDTreeFlann(pcd)

        estimate_surface_normals(
            self._pcd,
            voxel_down_sample_size=voxel_down_sample_size,
            normal_estimation_radius=normal_estimation_radius,
            orientation_ref=orientation_ref,
            inplace=True
        )

    def find_knn_radius(self, anchor: int, radius: float):
        num_points, idx, coordinates = self._kd_tree.search_radius_vector_3d(query=self._pcd.points[anchor],
                                                                             radius=radius)
        return idx

    def cluster_normals(self):

        for a in range(len(self.pcd.points)):
            r = 0.2
            k = 3
            pc_in_radius_idx = self.find_knn_radius(anchor=a, radius=r)
            model = KMeans(n_clusters=k, distance=CosineSimilarity, max_iter=100)
            pc_in_radius = torch.from_numpy(self.pcd.points[pc_in_radius_idx, : ])
            labels = model.fit_predict(pc_in_radius)

        """
        for each point cloud point we have:
            find the points with a radius using knn_radius
            find normals of those points using estimate_surface_normals
            cluster these normals
        """
        pass

    @property
    def pcd(self):
        return self._pcd
