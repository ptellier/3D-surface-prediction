from typing import Callable
import numpy as np
from numpy import ndarray
import open3d.geometry as o3d_geom
from open3d.geometry import PointCloud
from surface_normals import estimate_surface_normals


class ClusterNormals:

    def __init__(self, pcd: PointCloud, clustering_fn: Callable[[ndarray], float]):
        self.clustering_fn = clustering_fn
        self.pcd = pcd
        self.kd_tree = o3d_geom.KDTreeFlann(pcd)

        estimate_surface_normals(
            self.pcd,
            voxel_down_sample_size=0.01,
            normal_estimation_radius=0.02,
            orientation_ref=np.array([0.0, 0.0, 1.0]),
            inplace=True
        )

    def find_knn_radius(self, anchor: int, radius: float):
        num_points, idx, coordinates = self.kd_tree.search_radius_vector_3d(query=self.pcd.points[anchor], radius=radius)
        return idx

    def cluster_normals(self):
        """
        for each point cloud point we have:
            find the points with a radius using knn_radius
            find normals of those points using estimate_surface_normals
            cluster these normals
        """
        pass