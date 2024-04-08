"""
Main Script

Author: Yunhan Yang (yhyang.myron@gmail.com)
"""

import os
import cv2
import numpy as np
import open3d as o3d
import torch
import copy
import multiprocessing as mp
import pointops
import random
import argparse
import time
from segment_anything import build_sam, SamAutomaticMaskGenerator
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from PIL import Image
from os.path import join
from util import *

import pyviz3d.visualizer as viz


def plot_pcd_group_info_viz(pcd_dict, scene_name, save_dir):
    v = viz.Visualizer()
    coord = pcd_dict["coord"]
    color = pcd_dict["color"]
    group = pcd_dict["group"]
    color_coded = color.copy() * 0
    v.add_points("raw_pc", coord, color, point_size=30, visible=True)
    seg_idx = np.unique(group)
    for idx_mask in seg_idx:
        if idx_mask == -1:
            continue
        mask = (group==idx_mask).squeeze()          
        random_color = lambda: random.randint(0, 255)
        color_coded[mask,:] = torch.tensor([random_color(), random_color(), random_color()])
    v.add_points("seg_pc", coord, color_coded, point_size=30, visible=True)
    v.save(f'{save_dir}/viz/{scene_name}')


class G_Merging(object):

    """
    cuda acclerated merge two point cloud segmentation from different projection
    Author: Zhening Huang (zh340@cam.ac.uk)
    """

    def __init__(self, merge_grid=0.05, threshold=0.5):
        self.th = threshold
        self.merge_grid = merge_grid

    def __call__(self, pcd_1, pcd_2):
        pcd_1, pcd_2 = self.remove_unassigned(pcd_1), self.remove_unassigned(pcd_2)
        if pcd_1["coord"].shape[0] == 0 and pcd_2["coord"].shape[0] != 0:
            return pcd_2
        elif pcd_2["coord"].shape[0] == 0 and pcd_1["coord"].shape[0] != 0:
            return pcd_1
        elif pcd_1["coord"].shape[0] == 0 and pcd_2["coord"].shape[0] == 0:
            return None
        joint_pcd, mask_bin_1, mask_bin_2, group_1_id, group_2_id = self.pcd_concate(pcd_1, pcd_2)
        joint_pcd_cache = joint_pcd.copy()
        pcd_1_mask_reduced, pcd_2_mask_reduced = self.grid_downsample(joint_pcd, mask_bin_1, mask_bin_2)
        overlap_matrix = self.calculate_group_overlap(pcd_1_mask_reduced.cuda(), pcd_2_mask_reduced.cuda())
        pcd_merged = self.update_group(joint_pcd_cache, overlap_matrix, group_1_id, group_2_id)
        return pcd_merged

    def update_group(self, joint_pcd, overlap_matrix, group_1_id, group_2_id):
        # update group_1
        update_group_1 = torch.max(torch.from_numpy(overlap_matrix), dim=1)
        group_1_id_old = group_1_id.clone()
        group_1_id[update_group_1[0] > self.th] = group_2_id[
            update_group_1[1][update_group_1[0] > self.th]
        ]
        mapping_1 = dict(zip(group_1_id_old.cpu().numpy(), group_1_id.cpu().numpy()))
        for key, value in mapping_1.items():
            joint_pcd["group"][joint_pcd["group"] == key] = value

        # update group_2
        update_group_2 = torch.max(torch.from_numpy(overlap_matrix), dim=0)
        group_2_id_old = group_2_id.clone()
        group_2_id[update_group_2[0] > self.th] = group_1_id[
            update_group_2[1][update_group_2[0] > self.th]
        ]
        mapping_2 = dict(zip(group_2_id_old.cpu().numpy(), group_2_id.cpu().numpy()))
        for key, value in mapping_2.items():
            joint_pcd["group"][joint_pcd["group"] == key] = value

        return joint_pcd

    def pcd_concate(self, pcd_1, pcd_2):
        # rename pcd
        pcd_1, pcd_2 = self.rename_group(pcd_1, pcd_2)
        # concate group
        joint_pcd_group = np.concatenate((pcd_1["group"], pcd_2["group"]), axis=0)
        joint_pcd_coord = np.concatenate((pcd_1["coord"], pcd_2["coord"]), axis=0)
        joint_pcd_color = np.concatenate((pcd_1["color"], pcd_2["color"]), axis=0)
        joint_pcd = dict(
            coord=joint_pcd_coord, color=joint_pcd_color, group=joint_pcd_group
        )
        # get binary mask for groups in pcd_1 and pcd_2
        original_mask, group_id = self.group_to_mask(joint_pcd["group"])
        mask_bin_1_sparse = original_mask[:, group_id <= pcd_1["group"].max()].to_sparse()
        mask_bin_2_sparse = original_mask[:, group_id > pcd_1["group"].max()].to_sparse()
        group_id_1 = group_id[group_id <= pcd_1["group"].max()]
        group_id_2 = group_id[group_id > pcd_1["group"].max()]
        return joint_pcd, mask_bin_1_sparse, mask_bin_2_sparse, group_id_1, group_id_2

    def group_to_mask(self, group):
        orignal_mask_flat = torch.from_numpy(group).cuda().view(-1, 1)
        unique_groups = torch.unique(orignal_mask_flat)
        unique_groups = unique_groups[unique_groups != -1]
        original_mask = (orignal_mask_flat == unique_groups.view(1, -1)).float()
        return original_mask, unique_groups

    def rename_group(self, pcd_1, pcd_2):
        group_1 = pcd_1["group"]
        group_2 = pcd_2["group"]
        group_2 += group_1.max() + 1
        pcd_2["group"] = group_2
        return pcd_1, pcd_2

    def grid_downsample(self, joint_pcd, mask_bin_1, mask_bin_2):
        joint_pcd = self.downsample(joint_pcd)
        point_transition_matrix = self.voxel_transtion(joint_pcd)
        mask_bin_1_reduced = (
            torch.sparse.mm(point_transition_matrix, mask_bin_1).to_dense().cpu()
        )
        mask_bin_2_reduced = (
            torch.sparse.mm(point_transition_matrix, mask_bin_2).to_dense().cpu()
        )
        return mask_bin_1_reduced, mask_bin_2_reduced

    def voxel_transtion(self, joint_pcd):
        '''
        Create a sparse matrix for transition between original point and downsampled voxels.
        '''
        indices = torch.from_numpy(joint_pcd["inverse"]).cuda()
        column_idx = torch.arange(0, indices.shape[0]).cuda()
        indices_sparse_group = torch.vstack([column_idx, indices.squeeze()]).cuda()
        n_points = joint_pcd["coord"].shape[0]
        N_points = joint_pcd["inverse"].shape[0]
        point_transition_matrix = torch.sparse.FloatTensor(
            indices=indices_sparse_group,
            values=torch.ones(N_points).cuda(),
            size=(N_points, n_points),
        ).transpose(0, 1)
        return point_transition_matrix

    def downsample(self, data_dict):
        discrete_coord = np.floor(
            data_dict["coord"] / np.array(self.merge_grid)
        ).astype(np.int32)
        discrete_coord -= discrete_coord.min(0)
        key = self.fnv_hash_vec(discrete_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        idx_select = (
            np.cumsum(np.insert(count, 0, 0)[0:-1])
            + np.random.randint(0, count.max(), count.size) % count
        )
        idx_unique = idx_sort[idx_select]
        data_dict["coord"] = data_dict["coord"][idx_unique]
        data_dict["color"] = data_dict["color"][idx_unique]
        data_dict["group"] = data_dict["group"][idx_unique]
        data_dict["inverse"] = np.zeros_like(inverse)
        data_dict["inverse"][idx_sort] = inverse
        return data_dict

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr

    def calculate_group_overlap(self, mask_list_1, mask_list_2):
        """
        Calculate the IoU matrix between two lists of masks using PyTorch.

        Parameters:
        - mask_list_1: PyTorch tensor with shape (1000, n)
        - mask_list_2: PyTorch tensor with shape (1000, m)
        Returns:
        - iou_matrix: Matrix of IoU values with shape (n, m)
        #"""

        mask_list_1_t = mask_list_1.t()

        # Calculate intersection and union
        intersection = torch.mm(mask_list_1_t, mask_list_2)
        overlap_1 = (
            mask_list_1_t.sum(dim=1, keepdim=True)
            + mask_list_2.sum(dim=0, keepdim=True) * 0
        )
        overlap_2 = mask_list_1_t.sum(dim=1, keepdim=True) * 0 + mask_list_2.sum(
            dim=0, keepdim=True
        )
        union = torch.min(overlap_1, overlap_2)
        iou_matrix = intersection / union
        return iou_matrix.cpu().numpy()

    def remove_unassigned(self, pcd):
        mask = pcd["group"] != -1
        pcd["coord"] = pcd["coord"][mask]
        pcd["color"] = pcd["color"][mask]
        pcd["group"] = pcd["group"][mask]
        return pcd




def pcd_ensemble(org_path, new_path, data_path, vis_path):
    new_pcd = torch.load(new_path)
    new_pcd = num_to_natural(remove_small_group(new_pcd, 20))
    with open(org_path) as f:
        segments = json.load(f)
        org_pcd = np.array(segments['segIndices'])
    match_inds = [(i, i) for i in range(len(new_pcd))]
    new_group = cal_group(dict(group=new_pcd), dict(group=org_pcd), match_inds)
    print(new_group.shape)
    data = torch.load(data_path)
    visualize_partition(data["coord"], new_group, vis_path)


def get_sam(image, mask_generator):
    masks = mask_generator.generate(image)
    group_ids = np.full((image.shape[0], image.shape[1]), -1, dtype=int)
    num_masks = len(masks)
    group_counter = 0
    for i in reversed(range(num_masks)):
        # print(masks[i]["predicted_iou"])
        group_ids[masks[i]["segmentation"]] = group_counter
        group_counter += 1
    return group_ids


def get_pcd(scene_name, color_name, rgb_path, mask_generator, save_2dmask_path):
    intrinsic_path = join(rgb_path, scene_name, 'intrinsic', 'intrinsic_depth.txt')
    depth_intrinsic = np.loadtxt(intrinsic_path)

    pose = join(rgb_path, scene_name, 'pose', color_name[0:-4] + '.txt')
    depth = join(rgb_path, scene_name, 'depth', color_name[0:-4] + '.png')
    color = join(rgb_path, scene_name, 'color', color_name)

    depth_img = cv2.imread(depth, -1) # read 16bit grayscale image
    mask = (depth_img != 0)
    color_image = cv2.imread(color)
    color_image = cv2.resize(color_image, (640, 480))

    save_2dmask_path = join(save_2dmask_path, scene_name)
    if mask_generator is not None:
        group_ids = get_sam(color_image, mask_generator)
        if not os.path.exists(save_2dmask_path):
            os.makedirs(save_2dmask_path)
        img = Image.fromarray(num_to_natural(group_ids).astype(np.int16), mode='I;16')
        img.save(join(save_2dmask_path, color_name[0:-4] + '.png'))
    else:
        group_path = join(save_2dmask_path, color_name[0:-4] + '.png')
        img = Image.open(group_path)
        group_ids = np.array(img, dtype=np.int16)

    color_image = np.reshape(color_image[mask], [-1,3])
    group_ids = group_ids[mask]
    colors = np.zeros_like(color_image)
    colors[:,0] = color_image[:,2]
    colors[:,1] = color_image[:,1]
    colors[:,2] = color_image[:,0]

    pose = np.loadtxt(pose)
    
    depth_shift = 1000.0
    x,y = np.meshgrid(np.linspace(0,depth_img.shape[1]-1,depth_img.shape[1]), np.linspace(0,depth_img.shape[0]-1,depth_img.shape[0]))
    uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
    uv_depth[:,:,0] = x
    uv_depth[:,:,1] = y
    uv_depth[:,:,2] = depth_img/depth_shift
    uv_depth = np.reshape(uv_depth, [-1,3])
    uv_depth = uv_depth[np.where(uv_depth[:,2]!=0),:].squeeze()
    
    intrinsic_inv = np.linalg.inv(depth_intrinsic)
    fx = depth_intrinsic[0,0]
    fy = depth_intrinsic[1,1]
    cx = depth_intrinsic[0,2]
    cy = depth_intrinsic[1,2]
    bx = depth_intrinsic[0,3]
    by = depth_intrinsic[1,3]
    n = uv_depth.shape[0]
    points = np.ones((n,4))
    X = (uv_depth[:,0]-cx)*uv_depth[:,2]/fx + bx
    Y = (uv_depth[:,1]-cy)*uv_depth[:,2]/fy + by
    points[:,0] = X
    points[:,1] = Y
    points[:,2] = uv_depth[:,2]
    points_world = np.dot(points, np.transpose(pose))
    group_ids = num_to_natural(group_ids)
    save_dict = dict(coord=points_world[:,:3], color=colors, group=group_ids)
    return save_dict


def make_open3d_point_cloud(input_dict, voxelize, th):
    input_dict["group"] = remove_small_group(input_dict["group"], th)
    # input_dict = voxelize(input_dict)

    xyz = input_dict["coord"]
    if np.isnan(xyz).any():
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def cal_group(input_dict, new_input_dict, match_inds, ratio=0.5):
    group_0 = input_dict["group"]
    group_1 = new_input_dict["group"]
    group_1[group_1 != -1] += group_0.max() + 1
    
    unique_groups, group_0_counts = np.unique(group_0, return_counts=True)
    group_0_counts = dict(zip(unique_groups, group_0_counts))
    unique_groups, group_1_counts = np.unique(group_1, return_counts=True)
    group_1_counts = dict(zip(unique_groups, group_1_counts))

    # Calculate the group number correspondence of overlapping points
    group_overlap = {}
    for i, j in match_inds:
        group_i = group_1[i]
        group_j = group_0[j]
        if group_i == -1:
            group_1[i] = group_0[j]
            continue
        if group_j == -1:
            continue
        if group_i not in group_overlap:
            group_overlap[group_i] = {}
        if group_j not in group_overlap[group_i]:
            group_overlap[group_i][group_j] = 0
        group_overlap[group_i][group_j] += 1

    # Update group information for point cloud 1
    for group_i, overlap_count in group_overlap.items():
        # for group_j, count in overlap_count.items():
        max_index = np.argmax(np.array(list(overlap_count.values())))
        group_j = list(overlap_count.keys())[max_index]
        count = list(overlap_count.values())[max_index]
        total_count = min(group_0_counts[group_j], group_1_counts[group_i]).astype(np.float32)
        # print(count / total_count)
        if count / total_count >= ratio:
            group_1[group_1 == group_i] = group_j
    return group_1


def cal_2_scenes(pcd_list, index, voxel_size, voxelize, th=50):
    if len(index) == 1:
        return(pcd_list[index[0]])
    # print(index, flush=True)
    input_dict_0 = pcd_list[index[0]]
    input_dict_1 = pcd_list[index[1]]
    pcd0 = make_open3d_point_cloud(input_dict_0, voxelize, th)
    pcd1 = make_open3d_point_cloud(input_dict_1, voxelize, th)
    if pcd0 == None:
        if pcd1 == None:
            return None
        else:
            return input_dict_1
    elif pcd1 == None:
        return input_dict_0

    # Cal Dul-overlap
    pcd0_tree = o3d.geometry.KDTreeFlann(copy.deepcopy(pcd0))
    match_inds = get_matching_indices(pcd1, pcd0_tree, 1.5 * voxel_size, 1)
    pcd1_new_group = cal_group(input_dict_0, input_dict_1, match_inds)
    # print(pcd1_new_group)

    pcd1_tree = o3d.geometry.KDTreeFlann(copy.deepcopy(pcd1))
    match_inds = get_matching_indices(pcd0, pcd1_tree, 1.5 * voxel_size, 1)
    input_dict_1["group"] = pcd1_new_group
    pcd0_new_group = cal_group(input_dict_1, input_dict_0, match_inds)
    # print(pcd0_new_group)

    pcd_new_group = np.concatenate((pcd0_new_group, pcd1_new_group), axis=0)
    pcd_new_group = num_to_natural(pcd_new_group)
    pcd_new_coord = np.concatenate((input_dict_0["coord"], input_dict_1["coord"]), axis=0)
    pcd_new_color = np.concatenate((input_dict_0["color"], input_dict_1["color"]), axis=0)
    pcd_dict = dict(coord=pcd_new_coord, color=pcd_new_color, group=pcd_new_group)

    pcd_dict = voxelize(pcd_dict)
    return pcd_dict

def cal_2_scenes_speedup(pcd_list, index, voxel_size, voxelize):
    if len(index) == 1:
        return(pcd_list[index[0]])
    input_dict_0 = pcd_list[index[0]]
    input_dict_1 = pcd_list[index[1]]
    group_merge = G_Merging(merge_grid=voxel_size * 1.5, threshold=0.35)
    pcd_dict = group_merge(input_dict_0, input_dict_1)
    if pcd_dict is None:
        return None
    pcd_dict = voxelize(pcd_dict)
    return pcd_dict

def seg_pcd(scene_name, rgb_path, data_path, save_path, mask_generator, voxel_size, voxelize, th, train_scenes, val_scenes, save_2dmask_path):
    print(scene_name, flush=True)
    if os.path.exists(join(save_path, scene_name + ".pth")):
        return
    color_names = sorted(os.listdir(join(rgb_path, scene_name, 'color')), key=lambda a: int(os.path.basename(a).split('.')[0]))
    pcd_list = []
    for color_name in color_names:
        print(color_name, flush=True)
        pcd_dict = get_pcd(scene_name, color_name, rgb_path, mask_generator, save_2dmask_path)
        if len(pcd_dict["coord"]) == 0:
            continue
        pcd_dict = voxelize(pcd_dict)
        pcd_list.append(pcd_dict)
    
    while len(pcd_list) != 1:
        print(len(pcd_list), flush=True)
        new_pcd_list = []
        for indice in pairwise_indices(len(pcd_list)):
            # print(indice)
            # pcd_frame = cal_2_scenes(pcd_list, indice, voxel_size=voxel_size, voxelize=voxelize)
            pcd_frame = cal_2_scenes_speedup(pcd_list, indice, voxel_size=voxel_size, voxelize=voxelize)
            if pcd_frame is not None:
                new_pcd_list.append(pcd_frame)
        pcd_list = new_pcd_list
    seg_dict = pcd_list[0]
    seg_dict["group"] = num_to_natural(remove_small_group(seg_dict["group"], th))

    plot_pcd_group_info_viz(seg_dict, scene_name, "viz")

    torch.cuda.synchronize()
    finished_time = time.time()
    print(f"Time of {scene_name}: {finished_time - time_start}", flush=True)
    
    if scene_name in train_scenes:
        scene_path = join(data_path, "train", scene_name + ".pth")
    elif scene_name in val_scenes:
        scene_path = join(data_path, "val", scene_name + ".pth")
    data_dict = torch.load(scene_path)
    scene_coord = torch.tensor(data_dict["coord"]).cuda().contiguous()
    new_offset = torch.tensor(scene_coord.shape[0]).cuda()
    gen_coord = torch.tensor(seg_dict["coord"]).cuda().contiguous().float()
    offset = torch.tensor(gen_coord.shape[0]).cuda()
    gen_group = seg_dict["group"]
    indices, dis = pointops.knn_query(1, gen_coord, offset, scene_coord, new_offset)
    indices = indices.cpu().numpy()
    group = gen_group[indices.reshape(-1)].astype(np.int16)
    mask_dis = dis.reshape(-1).cpu().numpy() > 0.6
    group[mask_dis] = -1
    group = group.astype(np.int16)
    torch.save(num_to_natural(group), join(save_path, scene_name + ".pth"))


def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Segment Anything on ScanNet.')
    parser.add_argument('--rgb_path', type=str, help='the path of rgb data')
    parser.add_argument('--data_path', type=str, default='', help='the path of pointcload data')
    parser.add_argument('--save_path', type=str, help='Where to save the pcd results')
    parser.add_argument('--save_2dmask_path', type=str, default='', help='Where to save 2D segmentation result from SAM')
    parser.add_argument('--sam_checkpoint_path', type=str, default='', help='the path of checkpoint for SAM')
    parser.add_argument('--scannetv2_train_path', type=str, default='scannet-preprocess/meta_data/scannetv2_train.txt', help='the path of scannetv2_train.txt')
    parser.add_argument('--scannetv2_val_path', type=str, default='scannet-preprocess/meta_data/scannetv2_val.txt', help='the path of scannetv2_val.txt')
    parser.add_argument('--img_size', default=[640,480])
    parser.add_argument('--voxel_size', default=0.05)
    parser.add_argument('--th', default=50, help='threshold of ignoring small groups to avoid noise pixel')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    print(args)
    with open(args.scannetv2_train_path) as train_file:
        train_scenes = train_file.read().splitlines()
    with open(args.scannetv2_val_path) as val_file:
        val_scenes = val_file.read().splitlines()
    mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint=args.sam_checkpoint_path).to(device="cuda"))
    voxelize = Voxelize(voxel_size=args.voxel_size, mode="train", keys=("coord", "color", "group"))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    scene_names = sorted(os.listdir(args.rgb_path))
    for scene_name in scene_names:
        seg_pcd(scene_name, args.rgb_path, args.data_path, args.save_path, mask_generator, args.voxel_size, 
            voxelize, args.th, train_scenes, val_scenes, args.save_2dmask_path)
