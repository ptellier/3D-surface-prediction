import numpy as np
import open3d.geometry as o3d_geom
import surface_normals as sn


class ClusterNormals():

    def __init__(self, pcd):
        self.tree = o3d_geom.KDTreeFlann(pcd)
        self.normals = sn.estimate_surface_normals()
    

    def find_knn_radius(self, pcd: np.ndarray, anchor, radius: float):
        [k, idx, _] = self.tree.search_radius_vector_3d(pcd.points[anchor], radius)
        return idx

    #def cluster_normals():

        # for each point cloud data we have
        #    find the points with a radius using knn_radius
        #    find normals of those points using estimate_surface_normals
        #    cluster these normals
        # endfor