import copy
import numpy as np
import open3d.geometry as o3d_geom
from open3d.cpu.pybind.utility import IntVector


def estimate_surface_normals(
        o3d_pcd: o3d_geom.PointCloud,
        voxel_down_sample_size: float,
        normal_estimation_radius: float,
        orientation_ref: np.ndarray = np.array([0.0, 0.0, 1.0])
) -> tuple[o3d_geom.PointCloud, list[IntVector]]:
    """Estimate surface normals of an open3d Point Cloud. The point cloud is first down sampled.
    A voxel grid is created where each cube is the size of voxel_down_sample_size. All points that lie within the same cube are averaged into one point.
    The normal of each point is estimated using points within the given radius and then aligned to a reference orientation.

    Args:
        o3d_pcd (o3d_geom.PointCloud): an open 3d point cloud
        voxel_down_sample_size (float): voxel cube size for down sampling. None if no down sampling required.
        normal_estimation_radius (float): radius used to estimate normals
        orientation_ref (np.ndarray, optional): reference orientation for aligning normals. Defaults to np.array([0.0, 0.0, 1.0]).

    Returns:
        o3d_geom.PointCloud: an open3d point cloud with normals
        index_trace: list of vectors of indices that map from each downsampled point index  to its original point index.
    """

    o3d_pcd_estimation = copy.deepcopy(o3d_pcd)

    o3d_pcd_estimation, voxel_trace, index_trace = o3d_pcd.voxel_down_sample_and_trace(
        voxel_size=voxel_down_sample_size,
        min_bound=(-1, -1, -1),
        max_bound=(1, 1, 1)
    )
    o3d_pcd_estimation.estimate_normals(
        search_param=o3d_geom.KDTreeSearchParamRadius(radius=normal_estimation_radius)
    )
    o3d_pcd_estimation.orient_normals_to_align_with_direction(
        orientation_reference=orientation_ref
    )
    return o3d_pcd_estimation, index_trace
