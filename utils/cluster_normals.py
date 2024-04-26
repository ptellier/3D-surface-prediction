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

        self._pcd = estimate_surface_normals(
            self._pcd,
            voxel_down_sample_size=voxel_down_sample_size,
            normal_estimation_radius=normal_estimation_radius,
            orientation_ref=orientation_ref,
        )
        self._kd_tree = o3d_geom.KDTreeFlann(self._pcd)

    def find_knn_radius(self, anchor: int, radius: float):
        num_points, idx, coordinates = self._kd_tree.search_radius_vector_3d(query=self._pcd.points[anchor],
                                                                             radius=radius)
        return idx

    def cluster_normals(self, radius, k):

        for a in range(1):
            r = radius
            # k = k
            # don't use a k value of 1 since it throws an error saying k=1 is ambiguous
            k = [2,3] 
            bs = len(k)
            print(torch.tensor(k, dtype=torch.int64))
            pc_in_radius_idx = self.find_knn_radius(anchor=a, radius=r)
            model = KMeans(distance=CosineSimilarity, max_iter=100)

            # Normals of the points within the radius
            selected_points = np.asarray(self.pcd.normals)[pc_in_radius_idx]
            pc_in_radius = torch.from_numpy(np.tile((np.asarray(selected_points)), (bs, 1, 1)))


            result = model(x=pc_in_radius, k=torch.tensor(k))
            print(result.labels)
            print(result.inertia)



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
