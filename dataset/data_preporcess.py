import os
import cv2
import json
import pickle
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import multiprocessing as mp

# 相机内参（需根据实际数据集调整，若Meta中有内参可从文件读取）
DEFAULT_INTRINSICS = [
    [665.80768, 0.0, 637.642],
    [0.0, 665.80754, 367.56],
    [0.0, 0.0, 1.0]
]

OBJ_CLS = [7, 8]  # 需要过滤的类别（参考旧代码）
MIN_POINTS_PER_PART = 64  # 每个零件最小点云数量
MIN_NUM_PART = 2
MAX_NUM_PART = 20

def load_depth(img_path):
    """加载深度图（复用旧代码逻辑）"""
    depth_path = img_path + '_depth.png'
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    if len(depth.shape) == 3:
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        raise ValueError(f'Unsupported depth type for {depth_path}')
    return depth16


def load_meta(img_path):
    """加载Meta文件"""
    meta_path = img_path + '_meta.json'
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return meta


def backproject(depth, intrinsics, instance_mask):
    """反投影生成点云（复用旧代码逻辑）"""
    cam_fx = intrinsics[0, 0]
    cam_fy = intrinsics[1, 1]
    cam_cx = intrinsics[0, 2]
    cam_cy = intrinsics[1, 2]

    non_zero_mask = (depth > 0)
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)
    idxs = np.where(final_instance_mask)

    z = depth[idxs[0], idxs[1]] / 10000  # 若深度单位是mm，转m
    x = (idxs[1] - cam_cx) * z / cam_fx
    y = (idxs[0] - cam_cy) * z / cam_fy
    pts = np.stack((x, y, z), axis=1)
    return pts, idxs


def process_single_sample(base_path, save_dir):
    """处理单样本：生成点云+保存元信息"""
    # 1. 加载基础数据
    depth = load_depth(base_path)
    meta = load_meta(base_path)
    intrinsics = np.array(meta.get('intrinsic_matrix', DEFAULT_INTRINSICS))
    mask = cv2.imread(base_path + '_mask.png')[:, :, 2].astype(np.int32)
    h, w = mask.shape
    # 2. 提取有效实例
    all_inst_ids = sorted(list(np.unique(mask)))
    if all_inst_ids[0] == 0:
        del all_inst_ids[0]  # 移除背景
    if len(all_inst_ids) < MIN_NUM_PART or len(all_inst_ids) > MAX_NUM_PART:
        return None  # 过滤零件数不符合的样本
    
    # 3. 遍历实例生成点云
    part_pcs = []
    part_rotations = []  # 旋转矩阵
    part_translations = []  # 平移向量
    valid_inst_ids = []
    
    # for inst_id in all_inst_ids:
    #     # 生成实例mask
    #     inst_mask = (mask == (inst_id + 1))  # 还原背景偏移
    #     # 反投影生成点云
    #     pc, _ = backproject(depth, intrinsics, inst_mask)
    #     if len(pc) < MIN_POINTS_PER_PART:
    #         continue  # 过滤点云数量不足的实例
        
    #     # 从meta提取位姿（旋转/平移）
    #     # 需根据实际meta格式调整，此处参考旧代码逻辑
    #     inst_idx = meta['ins_indexes'].index(inst_id + 1)  # 还原背景偏移
    #     if meta['cls_indexes'][inst_idx] in OBJ_CLS:
    #         continue  # 过滤指定类别
        
    #     rot_mat = np.array(meta['rotation'][inst_idx])
    #     trans_vec = np.array(meta['translation'][inst_idx])
        
    #     part_pcs.append(pc.astype(np.float32))
    #     part_rotations.append(rot_mat.astype(np.float32))
    #     part_translations.append(trans_vec.astype(np.float32))
    #     valid_inst_ids.append(inst_id)

    raw_class_ids = meta['cls_indexes']
    raw_instance_ids = meta['ins_indexes']
    visibilities = np.array(meta['visibility'], dtype=np.float32)

    class_ids = []
    discard_masks = []
    valid_id = 0
    lowq_count = 0
    for raw_ins_id, cls_id, vis in zip(raw_instance_ids, raw_class_ids, visibilities):
        assert raw_ins_id > 0, f'instance ID 包含0, 请确认代码, file is {base_path}'
        if cls_id in OBJ_CLS:
            discard_masks.append(False)
            continue  # skip object instance
        # NOTE: mask的实例ID可能少于json中记录的ID。
        # 如[0,1,2] [0,1,2,3]，但json中的3不一定就是那个不存在mask的ID，可能是1和2中有可见性差的ID，所以要对齐。
        # 为此，设置了lowq_count计数器， 默认是0，但如果存在可见性太低的(<0.05?), 加1。
        # 比如，ID=1时几乎不可见，则后续 inst_id-1, 从[0,1,2,3]变为[0,0,1,2];
        # 又或者ID=2时几乎不可见，则后续 inst_id-1, 从[0,1,2,3]变为[0,1,0,2];
        # 又或者ID=1和2时都几乎不可见，则从[0,1,2,3]变为[0,0,0,1];
        ins_id = raw_ins_id - lowq_count
        # process foreground objects
        inst_mask = np.equal(mask, ins_id)
        # bounding box
        horizontal_indicies = np.where(np.any(inst_mask, axis=0))[0]
        vertical_indicies = np.where(np.any(inst_mask, axis=1))[0]
        if horizontal_indicies.shape[0] <= 0:
            lowq_count += 1
            discard_masks.append(False)
            continue
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
        # object occupies full image, rendering error, happens in CAMERA dataset
        if np.any(np.logical_or((x2 - x1) > w, (y2 - y1) > h)):
            lowq_count += 1
            discard_masks.append(False)
            continue
        # not enough valid depth observation
        final_mask = np.logical_and(inst_mask, depth > 0)
        if np.sum(final_mask) < 8 or vis <= 0.01:
            discard_masks.append(False)
            lowq_count += 1
            continue
        
        pc, _ = backproject(depth, intrinsics, final_mask)
        if len(pc) < MIN_POINTS_PER_PART:
            discard_masks.append(False)
            continue  # 过滤点云数量不足的实例

        # background objects and non-existing objects(实测下来可见性很低的在mask中不会被记录，但bbox保存了下来)
        # TODO: 由于不一致的可见性，导致bbox和mask的实例ID对不上
        if cls_id == 0 or (ins_id not in all_inst_ids):
            print(f'low q ins: {ins_id}, img: {base_path}, mask id: {all_inst_ids}')
            discard_masks.append(False)
            continue

        part_pcs.append(pc.astype(np.float32))
        discard_masks.append(True)
        valid_inst_ids.append(ins_id)
        class_ids.append(cls_id)
        valid_id += 1
    # no valid foreground objects
    if len(valid_inst_ids) == 0:
        return None

    discard_masks = np.array(discard_masks)

    ins_pose = np.array(meta['poses'], dtype=np.float32)
    # pose_to_self = np.linalg.inv(pose) @ pose
    if len(ins_pose.shape) == 2:
        ins_pose = ins_pose[None, :]
    # cls_ids = raw_class_ids
    obj_index = []
    fea_index = []
    for id in raw_class_ids:
        if id in [7, 8]:
            obj_index.append(True)
            fea_index.append(False)
        else:
            obj_index.append(False)
            fea_index.append(True)
    obj_pose = ins_pose[obj_index]
    if len(obj_pose) < 2:
        return None
    fea_pose = ins_pose[discard_masks]
    pose_to_self = []
    for i in range(obj_pose.shape[0]):
        obj_pose_i = obj_pose[i]
        pose_to_self.append(np.linalg.inv(obj_pose_i) @ fea_pose)
    match_pairs = []
    for i in range(pose_to_self[0].shape[0]):
        for j in range(pose_to_self[1].shape[0]):
            dis = (pose_to_self[0][i] - pose_to_self[1][j]).sum()
            if abs(dis) < 1e-6:
                match_pairs.append([valid_inst_ids[i], valid_inst_ids[j]])
    fea_odd = 0
    fea_dula = 0
    for id in class_ids:
        assert id not in [7, 8]
        if id % 2 == 1:
            fea_odd += 1
        else:
            fea_dula += 1

    if raw_class_ids[0] == 7:
        part_1_cls_ids = fea_odd
        part_2_cls_ids = fea_dula
    elif raw_class_ids[0] == 8:
        part_1_cls_ids = fea_dula
        part_2_cls_ids = fea_odd
    else:
        print(f"Not support this object {base_path}")
        return None
    part_rotations = np.array(meta['rotation'], dtype=np.float32)[obj_index]
    part_rotations = np.repeat(
        part_rotations, [part_1_cls_ids, part_2_cls_ids], axis=0)
    # part_rotations = ins_rots[discard_masks]
    
    # part_translations = np.array(meta['translation'], dtype=np.float32)[discard_masks]
    part_translations = np.array(meta['translation'], dtype=np.float32)[obj_index]
    part_translations = np.repeat(
        part_translations, [part_1_cls_ids, part_2_cls_ids], axis=0)

    # 过滤后零件数仍需符合要求
    num_parts = len(part_pcs)
    if num_parts < MIN_NUM_PART or num_parts > MAX_NUM_PART:
        return None
    
    # 4. 生成保存文件名
    # rel_path = os.path.relpath(base_path, save_dir).replace('/', '_')
    save_path = os.path.join(save_dir, f'{base_path}_pcs_meta.pkl')
    
    # 5. 构造保存字典
    save_data = {
        'part_pcs': part_pcs,  # 各零件点云 [P, N, 3]
        'part_rotations': part_rotations,  # 各零件旋转矩阵 [P, 3, 3]
        'part_translations': part_translations,  # 各零件平移向量 [P, 3]
        'num_parts': num_parts,
        'valid_inst_ids': valid_inst_ids,
        'match_pairs': match_pairs,  # 匹配对（若有）
        'fracture_label_threshold': 0.025  # 兼容原有字段
    }
    
    # 6. 保存数据
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return save_path


def batch_process_dataset(raw_data_dir, save_dir, split='train'):
    """批量处理数据集"""
    # 读取样本列表
    list_path = os.path.join(raw_data_dir, split, f'{split}_list_all.txt')
    with open(list_path, 'r') as f:
        img_paths = [os.path.join(raw_data_dir, split, line.strip()) for line in f.readlines()]
    save_dir = os.path.join(save_dir, split)
    os.makedirs(save_dir, exist_ok=True)
    num_workers = 8
    task_args = [(img_path, save_dir) for img_path in img_paths]
    with mp.Pool(num_workers) as p:
        save_paths = list(tqdm(p.starmap(process_single_sample, task_args), total=len(img_paths)))
    # 批量处理
    valid_sample_paths = []
    # for img_path in tqdm(img_paths):
    #     save_path = process_single_sample(img_path, save_dir)
    #     if save_path is not None:
    #         valid_sample_paths.append(save_path)
    for save_path in save_paths:
        if save_path is not None:
            valid_sample_paths.append(save_path)
    
    # 保存有效样本列表
    list_save_path = os.path.join(save_dir, f'{split}_valid_pcs_list.txt')
    with open(list_save_path, 'w') as f:
        for path in valid_sample_paths:
            f.write(f'{path}\n')
    print(f'Processed {len(valid_sample_paths)} valid samples for {split} split')


if __name__ == '__main__':
    RAW_DATA_DIR = '/data/yan/pose_dataset/multi_asm/NOCS'  # 原始数据集路径
    SAVE_DIR = '/data/yan/pose_dataset/multi_asm/NOCS'  # 预处理后保存路径
    
    # 处理训练集和验证集
    batch_process_dataset(RAW_DATA_DIR, SAVE_DIR, split='train')
    # batch_process_dataset(RAW_DATA_DIR, SAVE_DIR, split='test')
    # batch_process_dataset(RAW_DATA_DIR, SAVE_DIR, split='val')