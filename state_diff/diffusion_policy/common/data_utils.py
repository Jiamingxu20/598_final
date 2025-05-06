import diffusion_policy.common.data_utils
from typing import Optional
import time

import numpy as np
import h5py
import cv2
import torch
from tqdm import tqdm
from diffusion_policy.common.pcd_utils import (aggr_point_cloud_from_data, remove_plane_from_mesh,
    visualize_pcd)
from diffusion_policy.common.trans_utils import transform_to_world, transform_from_world, interpolate_poses
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
import cProfile
import pstats
import io



def save_dict_to_hdf5(dic, config_dict, filename, attr_dict=None):
    """
    ....
    """
    with h5py.File(filename, 'w') as h5file:
        if attr_dict is not None:
            for key, item in attr_dict.items():
                h5file.attrs[key] = item
        recursively_save_dict_contents_to_group(h5file, '/', dic, config_dict)

def recursively_save_dict_contents_to_group(h5file, path, dic, config_dict):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, np.ndarray):
            # h5file[path + key] = item
            if key not in config_dict:
                config_dict[key] = {}
            dset = h5file.create_dataset(path + key, shape=item.shape, **config_dict[key])
            dset[...] = item
        elif isinstance(item, dict):
            if key not in config_dict:
                config_dict[key] = {}
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item, config_dict[key])
        else:
            raise ValueError('Cannot save %s type'%type(item))

def load_dict_from_hdf5(filename):
    """
    ....
    """
    # with h5py.File(filename, 'r') as h5file:
    #     return recursively_load_dict_contents_from_group(h5file, '/')
    h5file = h5py.File(filename, 'r')
    return recursively_load_dict_contents_from_group(h5file, '/'), h5file

def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            # ans[key] = np.array(item)
            ans[key] = item
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

def modify_hdf5_from_dict(filename, dic):
    """
    Modify hdf5 file from a dictionary
    """
    with h5py.File(filename, 'r+') as h5file:
        recursively_modify_hdf5_from_dict(h5file, '/', dic)

def recursively_modify_hdf5_from_dict(h5file, path, dic):
    """
    Modify hdf5 file from a dictionary recursively
    """
    for key, item in dic.items():
        if isinstance(item, np.ndarray) and key in h5file[path]:
            h5file[path + key][...] = item
        elif isinstance(item, dict):
            recursively_modify_hdf5_from_dict(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot modify %s type'%type(item))

def _convert_actions(raw_actions, rotation_transformer, action_key):
    act_num, act_dim = raw_actions.shape
    is_bimanual = (act_dim == 14) or (act_dim == 12) 
    if is_bimanual:
        raw_actions = raw_actions.reshape(act_num * 2, act_dim // 2)
    
    if action_key == 'cartesian_action' or action_key == 'robot_eef_pose':
        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    elif action_key == 'joint_action':
        pass
    else:
        raise RuntimeError('unsupported action_key')
    if is_bimanual:
        proc_act_dim = raw_actions.shape[-1]
        raw_actions = raw_actions.reshape(act_num, proc_act_dim * 2)
    actions = raw_actions
    # vis_post_actions(actions[:,10:])
    return actions


def eef_action_to_6d(raw_actions):
    rotation_transformer = RotationTransformer(
        from_rep='rotation_6d', to_rep='axis_angle')
    act_num, act_dim = raw_actions.shape
    is_bimanual = (act_dim == 18) or (act_dim == 20)
    if is_bimanual:
        raw_actions = raw_actions.reshape(act_num * 2, act_dim // 2)
    
    pos = raw_actions[...,:3]
    rot = raw_actions[...,3:9]
    gripper = raw_actions[...,9:]
    rot = rotation_transformer.forward(rot)
    raw_actions = np.concatenate([
        pos, rot, gripper
    ], axis=-1).astype(np.float32)
    if is_bimanual:
        proc_act_dim = raw_actions.shape[-1]
        raw_actions = raw_actions.reshape(act_num, proc_act_dim * 2)
    actions = raw_actions
    # vis_post_actions(actions[:,10:])
    return actions

def policy_action_to_env_action(raw_actions, cur_eef_pose=None, action_mode='eef', num_bots=1):
    if action_mode == 'eef' and raw_actions.shape[1] == 3:
        assert cur_eef_pose is not None
        # raw_actions is of shape (T, 3). cur_eef_pose is of shape (6)
        # convert cur_eef_pose to (T, 6) and then concatenate with raw_actions
        raw_actions = np.concatenate([raw_actions, cur_eef_pose[None,3:].repeat(raw_actions.shape[0], axis=0)], axis=1)
        return raw_actions
    elif action_mode == 'eef':
        return eef_action_to_6d(raw_actions)
    elif action_mode == 'joint':
        return raw_actions
    
def point_cloud_proc(shape_meta, color_seq, depth_seq, extri_seq, intri_seq,
                  robot_base_pose_in_world_seq = None, qpos_seq=None, expected_labels=None, 
                  exclude_threshold=0.01, exclude_colors=[], teleop_robot=None, tool_names=[None], profile=False):
    # shape_meta: (dict) shape meta data for d3fields
    # color_seq: (np.ndarray) (T, V, H, W, C)
    # depth_seq: (np.ndarray) (T, V, H, W)
    # extri_seq: (np.ndarray) (T, V, 4, 4)
    # intri_seq: (np.ndarray) (T, V, 3, 3)
    # robot_name: (str) name of robot
    # meshes: (list) list of meshes
    # offsets: (list) list of offsets
    # finger_poses: (dict) dict of finger poses, mapping from finger name to (T, 6)
    # expected_labels: (list) list of expected labels
    if profile:
        # Initialize profiler
        pr = cProfile.Profile()
        pr.enable()


    boundaries = shape_meta['info']['boundaries']
    if boundaries == 'none':
        boundaries = None
    N_total = shape_meta['shape'][1]
    max_pts_num = shape_meta['shape'][1]
    
    resize_ratio = shape_meta['info']['resize_ratio']
    reference_frame = shape_meta['info']['reference_frame'] if 'reference_frame' in shape_meta['info'] else 'world'
    
    num_bots = 2 if robot_base_pose_in_world_seq.shape[2] == 8 else 1
    robot_base_pose_in_world_seq = robot_base_pose_in_world_seq.reshape(robot_base_pose_in_world_seq.shape[0], 4, num_bots, 4)
    robot_base_pose_in_world_seq = robot_base_pose_in_world_seq.transpose(0, 2, 1, 3)
    
    # TODO: pay attention to this part 
    # robot_base_in_world_temp = np.array([[[1.0, 0, 0.0, 0.80],
    #                                 [0, 1.0, 0.0, -0.22],
    #                                 [0.0, 0.0, 1.0, 0.03],
    #                                 [0.0, 0.0, 0.0, 1.0]],
    #                                 [[1.0, 0, 0.0, -0.515],
    #                                 [0, 1.0, 0.0, -0.22],
    #                                 [0.0, 0.0, 1.0, 0.03],
    #                                 [0.0, 0.0, 0.0, 1.0]],])
    # robot_base_pose_in_world_seq = np.repeat(robot_base_in_world_temp[None, ...], robot_base_pose_in_world_seq.shape[0], axis=0)

    H, W = color_seq.shape[2:4]
    resize_H = int(H * resize_ratio)
    resize_W = int(W * resize_ratio)
    
    new_color_seq = np.zeros((color_seq.shape[0], color_seq.shape[1], resize_H, resize_W, color_seq.shape[-1]), dtype=np.uint8)
    new_depth_seq = np.zeros((depth_seq.shape[0], depth_seq.shape[1], resize_H, resize_W), dtype=np.float32)
    new_intri_seq = np.zeros((intri_seq.shape[0], intri_seq.shape[1], 3, 3), dtype=np.float32)
    for t in range(color_seq.shape[0]):
        for v in range(color_seq.shape[1]):
            new_color_seq[t,v] = cv2.resize(color_seq[t,v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST)
            new_depth_seq[t,v] = cv2.resize(depth_seq[t,v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST)
            new_intri_seq[t,v] = intri_seq[t,v] * resize_ratio
            new_intri_seq[t,v,2,2] = 1.
    color_seq = new_color_seq
    depth_seq = new_depth_seq
    intri_seq = new_intri_seq
    T, V, H, W, C = color_seq.shape
    # assert H == 240 and W == 320 and C == 3
    aggr_src_pts_ls = []
    
    use_robot_pcd = shape_meta['info']['rob_pcd']
    use_tool_pcd = shape_meta['info']['eef_pcd']
    N_per_link = shape_meta['info']['N_per_link']
    N_eef = shape_meta['info']['N_eef']
    N_joints = shape_meta['info']['N_joints']
    voxel_size = shape_meta['info'].get('voxel_size', 0.02)
    remove_plane = shape_meta['info'].get('remove_plane', False)
    plane_dist_thresh = shape_meta['info'].get('plane_dist_thresh', 0.01)

    # Assigning a bright magenta to the robot point cloud (robot_pcd)
    # RGB for a bright magenta: [1.0, 0, 1.0]
    robot_color = np.array([1.0, 0, 1.0])  # Bright Magenta

    # Assigning a fluorescent green to the end-effector point cloud (ee_pcd)
    # RGB for a fluorescent green: [0.5, 1.0, 0]
    ee_color = shape_meta['info'].get('ee_color', [0.5, 1.0, 0])  # Fluorescent Green
    # ee_color = np.array([0.5, 1.0, 0])  # Fluorescent Green

    for t in range(T):
        curr_qpos = qpos_seq[t]
        qpos_dim = curr_qpos.shape[0] // num_bots
        
        robot_pcd_ls = []
        ee_pcd_ls = []
        

        for rob_i in range(num_bots):
            # transform robot pcd to world frame    
            robot_base_pose_in_world = robot_base_pose_in_world_seq[t, rob_i] if robot_base_pose_in_world_seq is not None else None

            tool_name = tool_names[rob_i]
            # compute robot pcd
            if use_robot_pcd:
                robot_pcd_pts = teleop_robot.get_robot_pcd(curr_qpos[qpos_dim*rob_i:qpos_dim*(rob_i+1)], N_joints=N_joints, N_per_link=N_per_link, out_o3d=False)
                robot_pcd_pts = (robot_base_pose_in_world @ np.concatenate([robot_pcd_pts, np.ones((robot_pcd_pts.shape[0], 1))], axis=-1).T).T[:, :3]

                robot_pcd_color = np.tile(robot_color, (robot_pcd_pts.shape[0], 1))
                robot_pcd_pts = np.concatenate([robot_pcd_pts, robot_pcd_color], axis=-1)
                robot_pcd_ls.append(robot_pcd_pts)

            if use_tool_pcd:
                ee_pcd_pts = teleop_robot.get_tool_pcd(curr_qpos[qpos_dim*rob_i:qpos_dim*(rob_i+1)], tool_name, N_per_inst=N_eef)    
                ee_pcd_pts = (robot_base_pose_in_world @ np.concatenate([ee_pcd_pts, np.ones((ee_pcd_pts.shape[0], 1))], axis=-1).T).T[:, :3]
            
                ee_pcd_color = np.tile(ee_color, (ee_pcd_pts.shape[0], 1))
                ee_pcd_pts = np.concatenate([ee_pcd_pts, ee_pcd_color], axis=-1)
                ee_pcd_ls.append(ee_pcd_pts)
                
        robot_pcd = np.concatenate(robot_pcd_ls , axis=0) if use_robot_pcd else None
        ee_pcd = np.concatenate(ee_pcd_ls, axis=0) if use_tool_pcd  else None

        aggr_src_pts = aggr_point_cloud_from_data(color_seq[t], depth_seq[t], intri_seq[t], extri_seq[t], 
                                                  downsample=True, N_total=N_total,
                                                  boundaries=boundaries, 
                                                  out_o3d=False, 
                                                  exclude_colors=exclude_colors,
                                                  voxel_size=voxel_size,
                                                  remove_plane=remove_plane,
                                                  plane_dist_thresh=plane_dist_thresh,)
        aggr_src_pts = np.concatenate(aggr_src_pts, axis=-1) # (N_total, 6), 6: (x, y, z, r, g, b)
        aggr_src_pts = np.concatenate([aggr_src_pts, robot_pcd], axis=0) if use_robot_pcd else aggr_src_pts
        aggr_src_pts = np.concatenate([aggr_src_pts, ee_pcd], axis=0) if use_tool_pcd else aggr_src_pts
        
        # transform to reference frame
        if reference_frame == 'world':
            pass
        elif reference_frame == 'robot':
            transformed_xyz = (np.linalg.inv(robot_base_pose_in_world_seq[t, 0]) @ np.concatenate([aggr_src_pts[:, :3], np.ones((aggr_src_pts.shape[0], 1))], axis=-1).T).T[:, :3]
            aggr_src_pts = np.concatenate([transformed_xyz, aggr_src_pts[:, 3:]], axis=1)
            
        aggr_src_pts_ls.append(aggr_src_pts.astype(np.float32))
            
        if profile:
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative' # or 'time'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats(10)
            print(s.getvalue())
 
    
    return aggr_src_pts_ls
