import numpy as np
import open3d as o3d
import cv2
import os
import sys
from math import atan2, cos, sin, sqrt, pi
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import concurrent
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from diffusion_policy.common.trans_utils import inverse_transform, rotation_matrix_zaxis, remove_scaling_from_transformation_matrix, apply_constraints_to_transformation
from diffusion_policy.common.cv2_util import visualize_hsv_channels
# import teaserpp_python




def intrinsics_dict2mat(intrinsics_dict):
    # intrinsics_dict: {'width': 640, 'height': 480, 'ppx': 320.0, 'ppy': 240.0, 'fx': 615.0, 'fy': 615.0, 'model': 'brown_conrady', 'coeffs': [0.0, 0.0, 0.0, 0.0, 0.0]}
    # intrinsics: (3, 3)
    intrinsics = np.zeros((3, 3))
    intrinsics[0, 0] = intrinsics_dict['fx']
    intrinsics[1, 1] = intrinsics_dict['fy']
    intrinsics[0, 2] = intrinsics_dict['ppx']
    intrinsics[1, 2] = intrinsics_dict['ppy']
    intrinsics[2, 2] = 1
    return intrinsics

def depth2fgpcd(depth, mask, cam_params):
    # depth: (h, w)
    # fgpcd: (n, 3)
    # mask: (h, w)
    
    # Get camera parameters
    fx, fy, cx, cy = cam_params

    # Ensure valid depth values are considered
    valid_mask = mask & (depth > 0)

    # Calculate point cloud coordinates using only valid masked values
    z_vals = depth[valid_mask]
    y_indices, x_indices = np.nonzero(valid_mask)  # Positions where mask is True
    x_vals = (x_indices - cx) * z_vals / fx
    y_vals = (y_indices - cy) * z_vals / fy

    # Combine x, y, z coordinates into a single array
    fgpcd = np.stack((x_vals, y_vals, z_vals), axis=-1)

    return fgpcd


def np2o3d(pcd, color=None):
    """
    Convert a NumPy array to an Open3D point cloud object. Optionally, assign colors to the points.

    :param pcd: NumPy array with shape (N, 6), where N is the number of points. The first three columns are XYZ coordinates, and the last three columns are normal vectors. If color is provided, the array should have shape (N, 3) for XYZ coordinates.
    :param color: Optional NumPy array with shape (N, 3) representing RGB colors for each point. Values should be in the range [0, 1].
    :return: Open3D point cloud object.
    """
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])

    if pcd.shape[1] == 6:
        pcd_o3d.normals = o3d.utility.Vector3dVector(pcd[:, 3:6])

    if color is not None:
        assert pcd.shape[0] == color.shape[0], "Point and color array must have the same number of points."
        assert color.max() <= 1, "Color values must be in the range [0, 1]."
        assert color.min() >= 0, "Color values must be in the range [0, 1]."
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)

    return pcd_o3d

def voxel_downsample(pcd, voxel_size, pcd_color=None):
    # :param pcd: [N,3] numpy array
    # :param voxel_size: float
    # :return: [M,3] numpy array
    pcd = np2o3d(pcd, pcd_color)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    if pcd_color is not None:
        return np.asarray(pcd_down.points), np.asarray(pcd_down.colors)
    else:
        return np.asarray(pcd_down.points)


def fps_downsample(pcd, N_total, pcd_color=None):
    pcd = np2o3d(pcd, pcd_color)
    pcd = pcd.farthest_point_down_sample(N_total)
    if pcd_color is not None:
        return np.asarray(pcd.points), np.asarray(pcd.colors)
    else:
        return np.asarray(pcd.points)

def color_seg(colors, color_name):
    """color segmentation

    Args:
        colors (np.ndaaray): (N, H, W, 3) RGB images in float32 ranging from 0 to 1
        color_name (string): color name of supported colors
    """
    color_name_to_seg_dict = {
        'yellow': {
            'h_min': 0.0,
            'h_max': 0.1,
            's_min': 0.2,
            's_max': 0.55,
            's_cond': 'and',
            'v_min': 0.8,
            'v_max': 1.01,
            'v_cond': 'or',
        },
        'black': {
            'h_min': 0.0,
            'h_max': 1.0,
            's_min': 0.0,
            's_max': 0.5,
            's_cond': 'and',
            'v_min': 0.0,
            'v_max': 0.45,
            'v_cond': 'and',
        }

    }
    
    colors_hsv = [cv2.cvtColor((color * 255).astype(np.uint8), cv2.COLOR_RGB2HSV) for color in colors]
    colors_hsv = np.stack(colors_hsv, axis=0)
    colors_hsv = colors_hsv / 255.
    seg_dict = color_name_to_seg_dict[color_name]
    h_min = seg_dict['h_min']
    h_max = seg_dict['h_max']
    s_min = seg_dict['s_min']
    s_max = seg_dict['s_max']
    s_cond = seg_dict['s_cond']
    v_min = seg_dict['v_min']
    v_max = seg_dict['v_max']
    v_cond = seg_dict['v_cond']
    mask = (colors_hsv[:, :, :, 0] > h_min) & (colors_hsv[:, :, :, 0] < h_max)
    if s_cond == 'and':
        mask = mask & (colors_hsv[:, :, :, 1] > s_min) & (colors_hsv[:, :, :, 1] < s_max)
    else:
        mask = mask | ((colors_hsv[:, :, :, 1] > s_min) & (colors_hsv[:, :, :, 1] < s_max))
    if v_cond == 'and':
        mask = mask & (colors_hsv[:, :, :, 2] > v_min) & (colors_hsv[:, :, :, 2] < v_max)
    else:
        mask = mask | ((colors_hsv[:, :, :, 2] > v_min) & (colors_hsv[:, :, :, 2] < v_max))
    return mask


def aggr_point_cloud_from_data(colors, depths, 
                               Ks, tf_world2cams, 
                               downsample=True, N_total=None,
                               masks=None, boundaries=None, 
                               out_o3d=True, exclude_colors=[],
                               voxel_size=0.02, remove_plane=False,
                               plane_dist_thresh=0.01):
    """
    aggregate point cloud data from multiple camera inputs.
    """
    N, H, W, _ = colors.shape
    
    pcds = []
    pcd_colors = []

    # Precompute the inverse transformations and color normalization factors
    inverse_transforms = [inverse_transform(tf) for tf in tf_world2cams]
    colors_normalized = colors / 255.0  # Normalize colors just once

    # Loop over each camera
    for i in range(N):
        depth = depths[i]
        mask = (depth > 0) & (depth < 2) if masks is None else masks[i] & (depth > 0)
        K = Ks[i]
        cam_param = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]  # fx, fy, cx, cy

        color = colors_normalized[i]

        if exclude_colors:
            for exclude_color in exclude_colors:
                mask &= (~color_seg(color[None], exclude_color))[0]
        
        color = color[mask]  # Apply mask to color early to reduce operations

        pcd = depth2fgpcd(depth, mask, cam_param)
        trans_pcd = (inverse_transforms[i] @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0))[:3, :].T

        if boundaries:
            x_lower, x_upper = boundaries['x_lower'], boundaries['x_upper']
            y_lower, y_upper = boundaries['y_lower'], boundaries['y_upper']
            z_lower, z_upper = boundaries['z_lower'], boundaries['z_upper']

            boundary_mask = ((trans_pcd[:, 0] > x_lower) & (trans_pcd[:, 0] < x_upper) &
                             (trans_pcd[:, 1] > y_lower) & (trans_pcd[:, 1] < y_upper) &
                             (trans_pcd[:, 2] > z_lower) & (trans_pcd[:, 2] < z_upper))
            trans_pcd = trans_pcd[boundary_mask]
            color = color[boundary_mask]

        pcds.append(trans_pcd)
        pcd_colors.append(color)

    # Concatenate all point clouds and colors
    pcds_pos = np.concatenate(pcds, axis=0)
    pcd_colors = np.concatenate(pcd_colors, axis=0)

    if out_o3d:
        pcd_o3d = np2o3d(pcds_pos, pcd_colors)
        if remove_plane:
            plane_model, inliers = pcd_o3d.segment_plane(distance_threshold=plane_dist_thresh,
                                                         ransac_n=3,
                                                         num_iterations=1000)
            pcd_o3d = pcd_o3d.select_by_index(inliers, invert=True)

        if downsample:
            pcd_o3d = pcd_o3d.voxel_down_sample(voxel_size)  # Adjust voxel size as needed
        if N_total is not None:
            pcd_o3d = pcd_o3d.farthest_point_down_sample(N_total)
        return pcd_o3d
    else:
        if remove_plane:
            pcd_o3d = np2o3d(pcds_pos, pcd_colors)
            plane_model, inliers = pcd_o3d.segment_plane(distance_threshold=plane_dist_thresh,
                                                         ransac_n=3,
                                                         num_iterations=1000)
            pcd_o3d = pcd_o3d.select_by_index(inliers, invert=True)
            pcds_pos = np.asarray(pcd_o3d.points)
            pcd_colors = np.asarray(pcd_o3d.colors)

        if downsample:
            pcds_pos, pcd_colors = voxel_downsample(pcds_pos, voxel_size, pcd_colors)  # Adjust voxel size as needed
        if N_total is not None:
            pcds_pos, pcd_colors = fps_downsample(pcds_pos, N_total, pcd_colors)

        return pcds_pos, pcd_colors

    
    

def get_color_mask(hsv_img, color):
    """
    Generates a mask for filtering a specified color in an HSV image.

    Args:
        hsv_img (numpy.ndarray): The HSV image to filter.
        color (str): The color to filter ('red', 'blue', or 'orange').

    Returns:
        numpy.ndarray: The mask for the specified color.
    """
    # Define color bounds in HSV space
    if color == 'red':
        lower_bounds = [np.array([0, 100, 100]), np.array([160, 100, 100])]
        upper_bounds = [np.array([5, 255, 255]), np.array([180, 255, 255])]
    elif color == 'blue':
        lower_bounds = [np.array([80, 100, 60])]
        upper_bounds = [np.array([110, 255, 255])]
    elif color == 'orange':
        lower_bounds = [np.array([0, 100, 100]), np.array([170, 100, 100])]
        upper_bounds = [np.array([15, 255, 255]), np.array([190, 255, 255])]
    else:
        raise ValueError("Unsupported color for masking.")

    # Create masks and merge them
    # mask = np.zeros_like(hsv_img[:, :, 0], dtype=bool)
    # for lower, upper in zip(lower_bounds, upper_bounds):
    #     color_mask = cv2.inRange(hsv_img, lower, upper)
    #     mask = np.logical_or(mask, color_mask.astype(bool))
        
    masks = [cv2.inRange(hsv_img, lower_bound, upper_bound) for lower_bound, upper_bound in zip(lower_bounds, upper_bounds)]
    return cv2.bitwise_or(*masks) if len(masks) > 1 else masks[0]
    # return mask.astype(np.uint8) * 255

def filter_point_cloud_by_color(pcd, color, color_space='rgb'):
    """
    Filters a point cloud to keep only points of a specific color.

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud.
        color (str): The color to keep ('red', 'blue', or 'orange').

    Returns:
        open3d.geometry.PointCloud: The filtered point cloud.
    """
    # Convert Open3D point cloud to NumPy arrays
    pcd_np = np.asarray(pcd.points)
    colors_np = (np.asarray(pcd.colors) * 255).astype(np.uint8)

    # Convert RGB to HSV and apply color mask
    hsv_colors = cv2.cvtColor(colors_np.reshape((1, -1, 3)), cv2.COLOR_RGB2HSV)
    mask = get_color_mask(hsv_colors, color)[0]

    # Filter the point cloud
    filtered_points = pcd_np[mask.astype(bool)]
    filtered_colors = colors_np[mask.astype(bool)] / 255.0

    # Create a new filtered point cloud
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    return filtered_pcd



    
def remove_plane_from_mesh(mesh, plane_z_value, tolerance=0.01):
    # Get triangle vertices
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    # Determine which triangles to keep (those not in the specified plane)
    triangles_to_keep = []
    for triangle in triangles:
        vertex_coords = vertices[triangle]
        if not np.all(np.abs(vertex_coords[:, 2] - plane_z_value) < tolerance):
            triangles_to_keep.append(triangle)

    # Create a new mesh from the remaining triangles
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles_to_keep))
    new_mesh.compute_vertex_normals()
    return new_mesh





def flip_all_inward_normals(pcd, center):
    # Get vertices and normals from the mesh
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    # Compute vectors from vertices to the center and normalize them
    norm_ref = points - center

    # Dot product between normals and vectors to center
    dot_products = np.einsum('ij,ij->i', norm_ref, normals)

    # Identify normals that need to be flipped
    flip_mask = dot_products < 0

    # Flip the normals
    normals[flip_mask] = -normals[flip_mask]

    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd

def segment_and_filp(pcd, visualize=False):
    labels = np.array(pcd.cluster_dbscan(eps=0.01, min_points=100))
    part_one = pcd.select_by_index(np.where(labels == 0)[0])
    part_two = pcd.select_by_index(np.where(labels > 0)[0])

    for part in [part_one, part_two]:
        if np.asarray(part.points).shape[0] > 0:
            # invalidate existing normals
            part.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  
            part.estimate_normals()
            part.orient_normals_consistent_tangent_plane(100)
            
            # get an accurate center
            # center = part.get_center()
            hull, _ = part.compute_convex_hull()
            center = hull.get_center()
            
            part = flip_all_inward_normals(part, center)
            
            if visualize:
                visualize_o3d([part, hull], title='part_normals', show_normal=True)
    
    return part_one + part_two




def visualize_o3d(geometry_list, title='O3D', view_point=None, point_size=5, pcd_color=[0, 0, 0],
    mesh_color=[0.5, 0.5, 0.5], show_normal=False, show_frame=True, path=''):
    vis = o3d.visualization.Visualizer()
    vis.create_window(title)
    types = []

    for geometry in geometry_list:
        type = geometry.get_geometry_type()
        # Point Cloud
        # if type == o3d.geometry.Geometry.Type.PointCloud:
        #     geometry.paint_uniform_color(pcd_color)
        # Triangle Mesh
        if type == o3d.geometry.Geometry.Type.TriangleMesh:
            geometry.paint_uniform_color(mesh_color)
        types.append(type)

        vis.add_geometry(geometry)
        vis.update_geometry(geometry)
    
    vis.get_render_option().background_color = np.array([0., 0., 0.])
    if show_frame:
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        vis.add_geometry(mesh)
        vis.update_geometry(mesh)

    if o3d.geometry.Geometry.Type.PointCloud in types:
        vis.get_render_option().point_size = point_size
        vis.get_render_option().point_show_normal = show_normal
    if o3d.geometry.Geometry.Type.TriangleMesh in types:
        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().mesh_show_wireframe = True

    vis.poll_events()
    vis.update_renderer()

    if view_point is None:
        vis.get_view_control().set_front(np.array([0.305, -0.463, 0.832]))
        vis.get_view_control().set_lookat(np.array([0.4, -0.1, 0.0]))
        vis.get_view_control().set_up(np.array([-0.560, 0.620, 0.550]))
        vis.get_view_control().set_zoom(0.2)
    else:
        vis.get_view_control().set_front(view_point['front'])
        vis.get_view_control().set_lookat(view_point['lookat'])
        vis.get_view_control().set_up(view_point['up'])
        vis.get_view_control().set_zoom(view_point['zoom'])

    # cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    # path = os.path.join(cd, '..', 'figures', 'images', f'{title}_{datetime.now().strftime("%b-%d-%H:%M:%S")}.png')

    if len(path) > 0:
        vis.capture_screen_image(path, True)
        vis.destroy_window()
    else:
        vis.run()
        
        

def load_template_pcd(asset_dir, mesh_name, tran, plane_z_value, mesh_scale_factor=0.001, flip_mesh=False, debug=False):
    scale_factor = mesh_scale_factor# 0.001  # Scaling down by a factor of 1000
    mesh_path = os.path.join(asset_dir, mesh_name)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.scale(scale_factor, center=np.array([0, 0, 0]))  # Scale relative to the origin
    mesh.transform(tran)

    if flip_mesh:
        mesh = remove_plane_from_mesh(mesh, -plane_z_value)
    else:
        mesh = remove_plane_from_mesh(mesh, plane_z_value)
        
    if debug:
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([mesh, coordinate_frame])

    number_points = 5000
    template_pcd = mesh.sample_points_poisson_disk(number_of_points=number_points)
    if debug:
        bounding_box = mesh.get_axis_aligned_bounding_box()
        extents = bounding_box.get_extent()

        print("Size of the object (extent in x, y, z):", extents)
        # Create a coordinate frame (you can adjust the size as needed)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        # Visualize the point cloud along with the coordinate frame
        o3d.visualization.draw_geometries([template_pcd, coordinate_frame])
    return template_pcd



# def perform_icp_and_get_results(source_pc, target_pc, detector, icp_iterations):
#     print('Matching...')
#     results = detector.match(target_pc, 0.5, 0.05)

#     print('Performing ICP...')
#     icp = cv2.ppf_match_3d_ICP(icp_iterations)
#     N = 2
#     _, results = icp.registerModelToScene(source_pc, target_pc, results[:N])
#     return results

def train_detector(template_pcd, preprocess_kwargs, voxel_size=0.005, debug=False):
    target_down, _ = preprocess_point_cloud(template_pcd, voxel_size, preprocess_kwargs=preprocess_kwargs, is_template=True, normal_only=True, debug=debug)
    pc = np.concatenate([np.asarray(target_down.points), np.asarray(target_down.normals)], axis=1).astype(np.float32)
    detector = cv2.ppf_match_3d_PPF3DDetector(relativeSamplingStep=0.015, relativeDistanceStep=0.1, numAngles=30)  # 0.025, 0.05
    detector.trainModel(pc)
    return detector, target_down


def perform_icp_and_get_results(source_pc, target_pc, detector, icp_iterations, prev_trans=np.eye(4), pose_prior=None, profile=False):
    # Preprocess the source point cloud and get the alignment transformation
    # source_pc_aligned, alignment_transform = initial_alignment_preprocessing(source_pc)
    # inverse_alignment_transform = np.linalg.inv(alignment_transform)

    if profile:
        start_time = time.time()
        print('Matching...')
    # Simplify conditional check
    use_detector = True  # The condition "True or (prev_trans is None)" always results in True
    if use_detector:
        start_matching = time.time() if profile else None

        results = detector.match(target_pc, 0.5, 0.1)  # 0.5, 0.05
        # acceptable_results = [result for result in results if is_pose_acceptable(result.pose)]
        # if not acceptable_results:
        #     print("No acceptable matches found.")
        #     return None
          # Apply constraints to the first two results
        [result.updatePose(remove_scaling_from_transformation_matrix(result.pose)) for result in results]
            # result.updatePose(apply_constraints_to_transformation(result.pose))

        end_matching = time.time() if profile else None
        if profile:
            print(f"Matching time: {end_matching - start_matching:.2f} seconds")
            print('Performing ICP...')
            start_icp = time.time()

    icp = cv2.ppf_match_3d_ICP(icp_iterations, numLevels=6) #, rejectionScale=0.5,
    if use_detector :
        _, results = icp.registerModelToScene(source_pc, target_pc, results)
        acceptable_results = [result for result in results if is_pose_acceptable(result.pose, **pose_prior)]
        if not acceptable_results and prev_trans is not None:
            trans_refined = prev_trans
        else:
            index_min = np.argmin([result.residual for result in acceptable_results])
            trans_refined = acceptable_results[index_min].pose

        # final_results = [result for result in acceptable_results if is_pose_acceptable(result.pose)]
        # if not acceptable_results:
        #     print("No acceptable matches found.")
        #     return None
        # index_min = np.argmin([result.residual for result in final_results[:2]])
        # trans_refined = final_results[index_min].pose
        # trans_refined = acceptable_results[0].pose if acceptable_results[0].residual < acceptable_results[1].residual else acceptable_results[1].pose

    else:
        centroid_src = np.mean(source_pc[:, :3], axis=0)
        centroid_tgt = np.mean(target_pc[:, :3], axis=0)
        dists_centroid = centroid_tgt - centroid_src
        
        trans_refined = prev_trans.copy()        
        trans_refined[:3, -1] += dists_centroid.reshape(-1)
        source_pc_copy = source_pc.copy()
        source_pc_copy[:, :3] += dists_centroid[None]
        
        retval, residual, pose = icp.registerModelToScene(source_pc_copy, target_pc)
        trans_refined = pose @ trans_refined

    end_icp = time.time() if profile else None
    if profile:
        print(f"ICP time: {end_icp - start_icp:.2f} seconds")
        total_time = time.time() - start_time
        print(f"Total time: {total_time:.2f} seconds")        
        
        
    # Combine the inverse alignment with the ICP result
    # trans_refined = np.dot(inverse_alignment_transform, trans_refined)
    return trans_refined


def preprocess_point_cloud(pcd, voxel_size, 
                           preprocess_kwargs,
                           normal_only=False, is_template=False, 
                           debug=False, fps=False):
    if fps==True:
        pcd_down = pcd.farthest_point_down_sample(100)
        pcd_inlier = pcd_down
    else:
        pcd_down = pcd.voxel_down_sample(voxel_size)

        outlier_kwargs = preprocess_kwargs['outlier_kwargs']
        # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=100,
        #                                                     std_ratio=4)
        # pcd_inlier = pcd.select_by_index(ind)
        # if debug:
        #     display_inlier_outlier(pcd, ind)
        if not is_template:
            cl, ind = pcd_down.remove_statistical_outlier(**outlier_kwargs) 
        # cl, ind = pcd_down.remove_radius_outlier(nb_points=50,
        #                                                     radius=0.05)
            pcd_inlier = pcd_down.select_by_index(ind)
            if debug:
                display_inlier_outlier(pcd_down, ind)

        else:
            pcd_inlier = pcd_down
        if debug:
            print(":: Downsample with a voxel size %.3f." % voxel_size)

    processed_pcd = pcd_inlier
    
    
    radius_normal = voxel_size * preprocess_kwargs['radius_normal_factor']
    if debug:
        print(":: Estimate normal with search radius %.3f." % radius_normal)
    processed_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    

    # processed_pcd.orient_normals_consistent_tangent_plane(100)
    orient_normals_consistent(processed_pcd, is_template=is_template, debug=debug)

        # points = np.asarray(processed_pcd.points)
        # centroid = compute_pca_center(points)
        # normals =  cv2.ppf_match_3d.computeNormalsPC3d(np.asarray(processed_pcd.points), NumNeighbors=30, FlipViewpoint=True, viewpoint=centroid)[1]
        # processed_pcd.normals = o3d.utility.Vector3dVector(normals[:, 3:])
        
    if debug:
        o3d.visualization.draw_geometries([processed_pcd], point_show_normal=True)
    
    if not normal_only:
        radius_feature = voxel_size * preprocess_kwargs['fpfh_feature_factor']
        if debug:
            print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            processed_pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    else:
        pcd_fpfh = None
    return processed_pcd, pcd_fpfh



def initial_alignment_preprocessing(source_pc, fixed_roll=180, fixed_pitch=0, fixed_z=0.0455):
    # Precompute the rotation matrix
    fixed_roll_rad = np.radians(fixed_roll)
    fixed_pitch_rad = np.radians(fixed_pitch)
    roll_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(fixed_roll_rad), -np.sin(fixed_roll_rad)],
        [0, np.sin(fixed_roll_rad), np.cos(fixed_roll_rad)]
    ])
    pitch_matrix = np.array([
        [np.cos(fixed_pitch_rad), 0, np.sin(fixed_pitch_rad)],
        [0, 1, 0],
        [-np.sin(fixed_pitch_rad), 0, np.cos(fixed_pitch_rad)]
    ])
    rotation_matrix = np.dot(pitch_matrix, roll_matrix)

    # Convert rotation and translation into a homogeneous transformation matrix
    alignment_transform = np.eye(4)
    alignment_transform[:3, :3] = rotation_matrix
    alignment_transform[2, 3] = fixed_z  # Apply fixed Z translation

    # Apply transformation
    homogeneous_source_pc = np.hstack([source_pc[:, :3], np.ones((source_pc.shape[0], 1))])
    aligned_source_pc = np.dot(homogeneous_source_pc, alignment_transform.T)[:, :3]

    return aligned_source_pc, alignment_transform




def is_pose_acceptable(pose, fixed_roll=180, fixed_pitch=0, fixed_z=0.055, tolerance_deg=15, tolerance_z=0.1):
    """
    Check if the pose is close to the given constraints.

    :param pose: The transformation matrix.
    :param fixed_roll: Fixed roll in degrees.
    :param fixed_pitch: Fixed pitch in degrees.
    :param fixed_z: Fixed Z translation.
    :param tolerance_deg: Tolerance in degrees for roll and pitch.
    :return: True if the pose is acceptable, False otherwise.
    """
    # Decompose transformation matrix
    rotation_matrix = pose[:3, :3]
    writable_matrix = np.array(rotation_matrix)
    euler_angles = R.from_matrix(writable_matrix).as_euler('xyz', degrees=True)
    # euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)

    roll, pitch, _ = euler_angles.flatten()

    # Normalize the angles to the range [0, 360)
    normalized_roll = roll % 360
    normalized_pitch = pitch % 360
    fixed_roll_normalized = fixed_roll % 360
    fixed_pitch_normalized = fixed_pitch % 360

    # Calculate the minimum angular difference
    def angular_difference(angle1, angle2):
        return min(abs(angle1 - angle2), 360 - abs(angle1 - angle2))

    roll_diff = angular_difference(normalized_roll, fixed_roll_normalized)
    pitch_diff = angular_difference(normalized_pitch, fixed_pitch_normalized)

    # Check against constraints

    return (roll_diff <= tolerance_deg and
            pitch_diff <= tolerance_deg and
            abs(pose[2, 3] - fixed_z) <= tolerance_z)

        
def compute_medoid_approx(points, subset_size=1000):
    if len(points) > subset_size:
        # Randomly sample a subset of points
        sampled_indices = np.random.choice(len(points), subset_size, replace=False)
        sampled_points = points[sampled_indices]
    else:
        sampled_points = points

    # Compute pairwise distances in a vectorized way
    distances = pairwise_distances(sampled_points)
    medoid_index = np.argmin(np.sum(distances, axis=0))

    return sampled_points[medoid_index]

def compute_medoid_ann(points, leaf_size=30):
    nbrs = NearestNeighbors(n_neighbors=len(points), algorithm='ball_tree', leaf_size=leaf_size).fit(points)
    distances, _ = nbrs.kneighbors(points)
    medoid_index = np.argmin(np.sum(distances, axis=1))

    return points[medoid_index]

def compute_medoid(points):
    distances = cdist(points, points, metric='euclidean')
    return points[np.argmin(np.sum(distances, axis=0))]
def compute_bounding_box_center(points):
    min_point = np.min(points, axis=0)
    max_point = np.max(points, axis=0)
    return (min_point + max_point) / 2
def compute_pca_center(points):
    pca = PCA(n_components=1)
    pca.fit(points)
    return pca.mean_

def orient_normals_consistent(pcd, center=None, center_method='pca', is_template=False, remove_outliers=False, nb_neighbors=20, std_ratio=2.0, debug=True):
    if remove_outliers:
        # Apply statistical outlier removal
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        pcd = pcd.select_by_index(ind)    
        
    # center choice: 'hull', 'robust', 'median', 'mean', 'pca'
    if is_template:
        center_method = None
    if center is not None:
        centroid = center
    else:
        if center_method=='robust':
            # Compute the center of the bounding box
            # bbox = pcd.get_oriented_bounding_box()
            bbox = pcd.get_axis_aligned_bounding_box()
            centroid = bbox.get_center()

            # # Compute the bounding boxes
            # oriented_bbox = pcd.get_oriented_bounding_box()
            # axis_aligned_bbox = pcd.get_axis_aligned_bounding_box()

            # # Set a distinct color for the bounding boxes
            # oriented_bbox.color = (1, 0, 0)  # Red color for oriented bbox
            # axis_aligned_bbox.color = (0, 1, 0)  # Green color for axis-aligned bbox

            # # Create small spheres at the centers of the bounding boxes
            # center_sphere_oriented = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            # center_sphere_oriented.translate(oriented_bbox.get_center())
            # center_sphere_oriented.paint_uniform_color((1, 0, 0))  # Red

            # center_sphere_axis_aligned = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            # center_sphere_axis_aligned.translate(axis_aligned_bbox.get_center())
            # center_sphere_axis_aligned.paint_uniform_color((0, 1, 0))  # Green

            # # Visualize
            # o3d.visualization.draw_geometries([pcd, oriented_bbox, axis_aligned_bbox, center_sphere_oriented, center_sphere_axis_aligned],
            #                                 window_name="Point Cloud with Bounding Boxes and Centers",
            #                                 width=800,
            #                                 height=600,
            #                                 left=50,
            #                                 top=50,
            #                                 point_show_normal=False,
            #                                 mesh_show_wireframe=False,
            #                                 mesh_show_back_face=False)
        elif center_method=='pca':
            points = np.asarray(pcd.points)
            centroid = compute_pca_center(points)

        elif center_method=='hull':
            hull, _ = pcd.compute_convex_hull()
            centroid = hull.get_center()
        elif center_method=='median':
            points = np.asarray(pcd.points)
            centroid = compute_medoid(points)
        else:
            # Compute the centroid of the point cloud
            centroid = np.mean(np.asarray(pcd.points), axis=0)
        if debug:
            center_vis = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            center_vis.translate(centroid)
            center_vis.paint_uniform_color((1, 0, 0))  # Red
            o3d.visualization.draw_geometries([pcd, center_vis], point_show_normal=True)


    # Orient normals towards the centroid
    # pcd.orient_normals_towards_camera_location(camera_location=centroid)
    pcd = flip_all_inward_normals(pcd=pcd, center=centroid)
    
def draw_registration_result(source, target, transformation):
    if isinstance(source, np.ndarray) and isinstance(target, np.ndarray):
        source_temp = np2o3d(source)
        target_temp = np2o3d(target)
    elif isinstance(source, o3d.geometry.PointCloud) and isinstance(target, o3d.geometry.PointCloud):   
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
    else:
        source_temp = o3d.geometry.PointCloud()
        source_temp.points = o3d.utility.Vector3dVector(source.points)
        target_temp = o3d.geometry.PointCloud()
        target_temp.points = o3d.utility.Vector3dVector(target.points)

    source_temp.paint_uniform_color([1, 0.706, 0])       # yellow
    target_temp.paint_uniform_color([0, 0.651, 0.929])   # blue
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    
    
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],)


    
def filter_pcd_with_mask(pcd, mask):
    """
    Filters a point cloud using a 2D mask.
    Assumes mask and point cloud are aligned and have the same FOV.
    """
    points = np.asarray(pcd)

    # Assuming the point cloud is already cropped to match the mask's FOV
    mask_indices = np.where(mask.flatten() > 0)[0]
    filtered_points = points[mask_indices]

    # Create a new point cloud object
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    return filtered_pcd


def extract_keypoint_from_img(colors, depths, intrinsics, extrinsics, boundaries, prev_trans_local_world, **keypoint_kwargs):
    raw_pcd = aggr_point_cloud_from_data(colors=colors,
                                    depths=depths,
                                    Ks=intrinsics,
                                    tf_world2cams=extrinsics,
                                    boundaries=boundaries,
                                    downsample=False)

    keypoints_world, all_trans_local_world = extract_keypoint_from_pcd(raw_pcd, 
                                                                       prev_trans_local_world=prev_trans_local_world, 
                                                                       **keypoint_kwargs)
    
    return keypoints_world, all_trans_local_world

def extract_keypoint_from_pcd(raw_pcd, obj_points_local_3d, template_pcds, 
                              detectors, voxel_size, prev_trans_local_world, 
                              preprocess_kwargs,
                              objects_color=['orange', 'blue'], 
                              max_z_value = 0.061,
                              pose_prior=None,
                              debug=False):   
    # remove the table 
    if debug:
        o3d.visualization.draw_geometries([raw_pcd])
    plane_model, inliers = raw_pcd.segment_plane(distance_threshold=0.011, # 0.009 works well
                                            ransac_n=3,
                                            num_iterations=100)

    pcd_wo_table = raw_pcd.select_by_index(inliers, invert=True)
    if debug:
        o3d.visualization.draw_geometries([pcd_wo_table])
            
    # Filter pcd_wo_table based on the z-value
    pcd_wo_table_points = np.asarray(pcd_wo_table.points)
    below_or_equal_max_z = pcd_wo_table_points[:, 2] <= max_z_value+0.002
    filtered_points = pcd_wo_table_points[below_or_equal_max_z]

    # Creating a new point cloud with the filtered points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # Optional: If you also want to keep the color or normals (if they exist)
    if pcd_wo_table.has_colors():
        filtered_colors = np.asarray(pcd_wo_table.colors)[below_or_equal_max_z]
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    if debug:
        o3d.visualization.draw_geometries([filtered_pcd])


    L0_pcd = filter_point_cloud_by_color(filtered_pcd, objects_color[0])
    L1_pcd = filter_point_cloud_by_color(filtered_pcd, objects_color[1])
    if debug:
        o3d.visualization.draw_geometries([L0_pcd, L1_pcd])

    all_keypoints_world, all_trans_local_world = icp(pcd_list=[L0_pcd, L1_pcd], 
                                                     template_pcds=template_pcds,
                                                     detectors=detectors, 
                                                     keypoints_local_3d=obj_points_local_3d, 
                                                     preprocess_kwargs=preprocess_kwargs,
                                                     voxel_size=voxel_size,
                                                     prev_trans_local_world=prev_trans_local_world, 
                                                     pose_prior=pose_prior,
                                                     debug=debug)

    return all_keypoints_world, all_trans_local_world


def icp(pcd_list, template_pcds, detectors, keypoints_local_3d, 
        preprocess_kwargs,
        method = 'surface_matching', 
        voxel_size=0.005, 
        prev_trans_local_world=None, 
        pose_prior=None,
        debug=False):

    # method: fast_global_reg, multi_init, surface_matching
    all_keypoints_world = []
    surface_point_only = False
    all_trans_local_world = []
    for i , template_pcd in enumerate(template_pcds):      

        if pcd_list is not None:
            curr_pcd = pcd_list[i]

        if debug:
            o3d.visualization.draw_geometries([curr_pcd, template_pcd])

        if method == 'fast_global_reg':
            source_down, source_fpfh = preprocess_point_cloud(curr_pcd, voxel_size, preprocess_kwargs=preprocess_kwargs, is_template=False, debug=debug)
            target_down, target_fpfh = preprocess_point_cloud(template_pcd , voxel_size, preprocess_kwargs=preprocess_kwargs,is_template=True, debug=debug)

            if debug:
                o3d.visualization.draw_geometries([source_down, target_down])
            acceptable_pose_found = False
            attempt = 0
            max_attempts=15
            while not acceptable_pose_found and attempt < max_attempts:
                result_ransac = execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
                trans_local_world = inverse_transform(result_ransac.transformation)

                if is_pose_acceptable(trans_local_world, tolerance_deg=5, tolerance_z=0.01):
                    acceptable_pose_found = True
                    if debug:
                        print(f"Acceptable global registration transformation L{i} found on attempt {attempt}: {trans_local_world}")
                        draw_registration_result(source_down, target_down, result_ransac.transformation)
                else:
                    attempt += 1
            if not acceptable_pose_found:
                trans_local_world = prev_trans_local_world[i]

        elif method == 'multi_init':    
            if surface_point_only == True:
                flat_template_pcd = np.asarray(template_pcd.points)
                flat_template_pcd[:, 2] = 0
                template_pcd.points = o3d.utility.Vector3dVector(flat_template_pcd)
                flat_curr_pcd = np.asarray(curr_pcd.points)
                flat_curr_pcd[:, 2] = 0
                curr_pcd.points = o3d.utility.Vector3dVector(flat_curr_pcd)                   
            source_down, source_fpfh = preprocess_point_cloud(curr_pcd, voxel_size, preprocess_kwargs=preprocess_kwargs, is_template=False, debug=debug)
            target_down, target_fpfh = preprocess_point_cloud(template_pcd , voxel_size, preprocess_kwargs=preprocess_kwargs,is_template=True, debug=debug)
            # o3d.visualization.draw_geometries([source_down, target_down])

            # Multiple initializations
            transformation_init = [rotation_matrix_zaxis(deg) for deg in range(0, 360, 20)]
            threshold = 100  # Adjust this threshold as needed

            # Run ICP in parallel
            results = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(execute_icp, source_down, target_down, threshold, trans) for trans in transformation_init]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
                    
            acceptable_results = [result for result in results if is_pose_acceptable(result.transformation)]

            # Find the best result (e.g., based on fitness)
            best_fit = max(acceptable_results, key=lambda icp_result: icp_result.fitness)
            trans_local_world = inverse_transform(best_fit.transformation)

            # Visualize the best result
            # draw_registration_result(source_down, target_down, best_transformation)
        elif method == 'surface_matching':
            scene_down, scene_fpfh = preprocess_point_cloud(curr_pcd, voxel_size,preprocess_kwargs=preprocess_kwargs, normal_only=True,  is_template=False, debug=debug)
            target_down, target_fpfh = preprocess_point_cloud(template_pcd , voxel_size, preprocess_kwargs=preprocess_kwargs, normal_only=True, is_template=True, debug=debug)

            detector = detectors[i]
            pc = np.concatenate([np.asarray(target_down.points), np.asarray(target_down.normals)], axis=1).astype(np.float32)
            pcTest = np.concatenate([np.asarray(scene_down.points), np.asarray(scene_down.normals)], axis=1).astype(np.float32)
            pose = perform_icp_and_get_results(pc, pcTest, detector, 
                                               icp_iterations=100, 
                                               prev_trans=prev_trans_local_world[i], 
                                               pose_prior=pose_prior)
            if debug:
                pct = cv2.ppf_match_3d.transformPCPose(pc, pose)

                pct_pcd = o3d.geometry.PointCloud()
                pct_pcd.points = o3d.utility.Vector3dVector(pct[:, :3])
                pct_pcd.paint_uniform_color([1, 0, 0])  # Red color

                # Create Open3D point cloud for 'pcTest'
                pcTest_pcd = o3d.geometry.PointCloud()
                pcTest_pcd.points = o3d.utility.Vector3dVector(pcTest[:, :3])
                pcTest_pcd.paint_uniform_color([0, 1, 0])  # Green color

                # Visualize the two point clouds together
                o3d.visualization.draw_geometries([pct_pcd, pcTest_pcd])
                
            trans_local_world = pose  
        elif method == '2d_icp':
            scene_down, scene_fpfh = preprocess_point_cloud(curr_pcd, voxel_size,preprocess_kwargs=preprocess_kwargs, normal_only=True,  is_template=False, debug=debug)
            target_down, target_fpfh = preprocess_point_cloud(template_pcd , voxel_size, preprocess_kwargs=preprocess_kwargs, normal_only=True, is_template=True, debug=debug)

            threshold = 1
            data = Point_cloud()
            ref = Point_cloud()
            ref.init_from_points(np.asarray(scene_down.points))
            data.init_from_points(np.asarray(target_down.points))
            icp_method = 'point2point'
            rotation, translation, rms_list = ICP(data, ref, method=icp_method, 
                                 exclusion_radius=threshold, sampling_limit=None, 
                                 verbose=False,
                                 pose_prior=pose_prior)

            # Visualize the registration result
            draw_registration_result(data, ref, np.vstack((np.hstack((rotation, translation.reshape(3, 1))), np.array([0, 0, 0, 1]))) )
        elif method == 'teaserpp':
            # Populating the parameters
            solver_params = teaserpp_python.RobustRegistrationSolver.Params()
            solver_params.cbar2 = 1
            solver_params.noise_bound = 0.05
            solver_params.estimate_scaling = False
            solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
            solver_params.rotation_gnc_factor = 1.4
            solver_params.rotation_max_iterations = 100
            solver_params.rotation_cost_threshold = 1e-12
            solver = teaserpp_python.RobustRegistrationSolver(solver_params)

            scene_down, scene_fpfh = preprocess_point_cloud(curr_pcd, voxel_size,preprocess_kwargs=preprocess_kwargs, normal_only=True,  is_template=False, debug=debug)
            target_down, target_fpfh = preprocess_point_cloud(template_pcd , voxel_size, preprocess_kwargs=preprocess_kwargs, normal_only=True, is_template=True, debug=debug)

            solver.solve(np.asarray(target_down.points).transpose(), np.asarray(scene_down.points).transpose())
            solution = solver.getSolution()
            pose = np.vstack((np.hstack((solution.rotation, solution.translation.reshape(3, 1))), np.array([0, 0, 0, 1])))
            draw_registration_result(target_down, scene_down, pose )


        # trans_local_world is 4x4 transformation matrix
        # and keypoints_local_3d[i] is a Nx3 array of 3D points
        all_trans_local_world.append(trans_local_world)
        
        # Step 1: Convert to homogeneous coordinates
        homogeneous_keypoints = np.ones((keypoints_local_3d[i].shape[0], 4))
        homogeneous_keypoints[:, :3] = keypoints_local_3d[i]
        
        # Step 2: Perform the matrix multiplication
        transformed_homogeneous_keypoints = trans_local_world.dot(homogeneous_keypoints.T).T

        # Step 3: Convert back to 3D coordinates
        keypoints_world = transformed_homogeneous_keypoints[:, :3]
        
        all_keypoints_world.append(keypoints_world)
                     
    return all_keypoints_world, all_trans_local_world


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size, debug=False):
    distance_threshold = voxel_size * 0.5
    if debug:
        print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance = distance_threshold))
    return result

def execute_icp(source, target, threshold, init_trans):
    return o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())




def generate_initial_transformations(n, roll=180, pitch=0, z=0.06, x_range=(-1, 1), y_range=(-1, 1)):
    """
    Generate multiple initial transformations.
    
    :param n: Number of initial transformations to generate.
    :param roll: Fixed roll value in degrees.
    :param pitch: Fixed pitch value in degrees.
    :param z: Fixed z translation.
    :param x_range: Range of x translations as a tuple (min, max).
    :param y_range: Range of y translations as a tuple (min, max).
    :return: List of initial transformation matrices.
    """
    transformations = []
    roll_rad = np.deg2rad(roll)
    pitch_rad = np.deg2rad(pitch)

    for _ in range(n):
        x_trans = np.random.uniform(*x_range)
        y_trans = np.random.uniform(*y_range)

        # Create rotation matrix using roll and pitch
        rot_x = np.array([[1, 0, 0],
                          [0, np.cos(roll_rad), -np.sin(roll_rad)],
                          [0, np.sin(roll_rad), np.cos(roll_rad)]])
        rot_y = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                          [0, 1, 0],
                          [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
        rot_matrix = np.dot(rot_y, rot_x)

        # Create transformation matrix
        trans_matrix = np.eye(4)
        trans_matrix[:3, :3] = rot_matrix
        trans_matrix[0, 3] = x_trans
        trans_matrix[1, 3] = y_trans
        trans_matrix[2, 3] = z

        transformations.append(trans_matrix)
    
    return transformations


def visualize_pcd(points):
    print("Processing input for visualization...")
    
    if isinstance(points, tuple):
        print("Input type: Tuple (xyz, rgb)")
        xyz, rgb = points[0], points[1]
        if np.max(rgb) > 1.0:
            rgb = rgb / 255.0
        print(f"XYZ shape: {xyz.shape}, RGB shape: {rgb.shape}")
    elif isinstance(points, np.ndarray):
        print("Input type: NumPy array")
        if points.shape[0] == 3 or points.shape[0] == 6:
            xyz = points[:3, :].T
            rgb = points[3:6, :].T  if points.shape[0] == 6 else None
        elif points.shape[1] == 3 or points.shape[1] == 6:
            xyz = points[:, :3]
            rgb = points[:, 3:6] if points.shape[1] == 6 else None
        else:
            raise ValueError("Array shape not recognized. Expected (N, 3), (3, N), (N, 6), or (6, N).")
        print(f"XYZ shape: {xyz.shape}, RGB shape: {'N/A' if rgb is None else rgb.shape}")
    elif isinstance(points, o3d.geometry.PointCloud):
        print("Input type: Open3D PointCloud")
        pcd = points
        o3d.visualization.draw_geometries([pcd])
        return
    else:
        raise ValueError("Invalid input type. Must be a NumPy array, tuple, or Open3D point cloud object.")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    print(f"Visualizing point cloud with {len(pcd.points)} points.")
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, coordinate_frame])
