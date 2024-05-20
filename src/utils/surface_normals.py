import copy
import numpy as np
import open3d.geometry as o3d_geom
import open3d.utility as o3d_util
from numpy.linalg import norm
from open3d.cpu.pybind.utility import IntVector


def pcd_img_to_o3d_pcd(
        pcd: np.ndarray, rgb_img: np.ndarray = None
) -> o3d_geom.PointCloud:
    """Creates an o3d_geom.PointCloud object.
        Overlays rgb image if specified.

    Args:
        pcd (np.array): an (h, w, 3) array where h is the height and w is the width of the point cloud
        img (np.array): an (h, w, c) array where h is the height and w is the width of the image

    Returns:
        o3d_geom.PointCloud : an open3d point cloud
    """
    pcd = pcd.reshape((pcd.shape[0] * pcd.shape[1]), pcd.shape[2])
    non_nan_idx = ~np.isnan(pcd).any(axis=1)
    pcd_clean = pcd[non_nan_idx]

    o3d_pcd = o3d_geom.PointCloud()
    o3d_pcd.points = o3d_util.Vector3dVector(pcd_clean)

    if rgb_img is not None:
        # convert color to between 0.0 to 1.0
        rgb_img = rgb_img.reshape(
            (rgb_img.shape[0] * rgb_img.shape[1]), rgb_img.shape[2]
        )
        rgb_img = rgb_img[non_nan_idx]
        rgb_img = rgb_img.astype(np.float64)
        rgb_img /= 255.0
        rgb_img = rgb_img[:, 0:3].reshape(-1, 3)
        o3d_pcd.colors = o3d_util.Vector3dVector(rgb_img)

    return o3d_pcd


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


def color_o3d_pcd_by_normals(o3d_pcd: o3d_geom.PointCloud) -> o3d_geom.PointCloud:
    """Colors an open3d point cloud by its normals. Assumes that a point cloud has normals.

    Args:
        o3d_pcd (o3d_geom.PointCloud): an open3d point cloud with normals

    Returns:
        o3d_geom.PointCloud: and open3d point cloud colored by its normals
    """
    normals = np.asarray(o3d_pcd.normals)
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([0, 0, 1])
    r = np.dot(normals, v1) / (norm(normals, axis=1) * norm(v1))
    g = np.dot(normals, v2) / (norm(normals, axis=1) * norm(v2))
    b = np.dot(normals, v3) / (norm(normals, axis=1) * norm(v3))
    np_colors = np.vstack([r, g, b]).T
    o3d_pcd.colors = o3d_util.Vector3dVector(np_colors)
    return o3d_pcd
