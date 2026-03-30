# # This is for the APE dataset. The pointcloud is stored in a .pkl file.
# # The .pkl file contains a list of dictionaries. Each dictionary contains the following keys:
# # 'pts': Nx3 numpy array of pointcloud points
# # 'rand_rotation'
# # 'rgb'
# # 'pts_raw' # N*3
# # 'rgb_raw'
# # 'choose'
# # 'mask'
# # 'center'
# # 'category_label'
# # 'pred_class_ids'
# # 'pred_bboxes'
# # 'pred_scores'

import os
import pickle
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader


class AllPieceMatchingDataset(Dataset):
    """适配自定义数据集的几何零件装配数据集（基于RGB/Mask/Depth反投影点云）"""
    def __init__(
            self,
            data_dir,  # 预处理后的数据目录
            data_fn,   # 有效样本列表文件（train/val_valid_pcs_list.txt）
            category="",  # 兼容原有参数，暂未使用
            num_points=1000,  # 每个样本总采样点数
            min_num_part=2,
            max_num_part=20,
            shuffle_parts=False,
            rot_range=-1,
            overfit=-1,
            length=-1,
            fracture_label_threshold=0.025,
    ):
        # 保留核心参数
        self.data_dir = data_dir
        self.num_points = num_points
        self.min_num_part = min_num_part
        self.max_num_part = max_num_part
        self.shuffle_parts = shuffle_parts
        self.rot_range = rot_range
        self.fracture_label_threshold = fracture_label_threshold

        # 读取预处理后的样本列表
        self.data_list = self._read_data(data_fn)
        print(f"Dataset length (raw): {len(self.data_list)}")

        # 过滤/裁剪样本（兼容原有逻辑）
        if overfit > 0:
            self.data_list = self.data_list[:overfit]
        if 0 < length < len(self.data_list):
            self.length = length
            if shuffle_parts:
                random.shuffle(self.data_list)
                self.data_list = self.data_list[:length]
        else:
            self.length = len(self.data_list)
        print(f"Dataset length (final): {self.length}")

    def __len__(self):
        return self.length

    def _read_data(self, data_fn):
        """读取预处理后的有效样本列表"""
        # 缓存逻辑（兼容原有）
        pre_compute_file_name = f"custom_pcs_meta_list_{self.min_num_part}_{self.max_num_part}_{os.path.basename(data_fn)}"
        cache_path = os.path.join(self.data_dir, pre_compute_file_name)
        
        # 加载缓存
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # 读取样本列表文件
        with open(os.path.join(self.data_dir, data_fn), 'r') as f:
            data_list = [line.strip() for line in f.readlines()]
        
        # 过滤存在的文件
        data_list = [p for p in data_list if os.path.exists(p)]
        
        # 缓存列表
        with open(cache_path, 'wb') as f:
            pickle.dump(data_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        return data_list

    @staticmethod
    def _recenter_pc(pc):
        """点云中心化（复用原有逻辑）"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid[None]
        return pc, centroid

    def _rotate_pc(self, pc):
        """点云旋转（复用原有逻辑）"""
        if self.rot_range > 0.0:
            rot_euler = (np.random.rand(3) - 0.5) * 2.0 * self.rot_range
            rot_mat = R.from_euler("xyz", rot_euler, degrees=True).as_matrix()
        else:
            rot_mat = R.random().as_matrix()
        pc = (rot_mat @ pc.T).T
        quat_gt = R.from_matrix(rot_mat.T).as_quat()
        quat_gt = quat_gt[[3, 0, 1, 2]]  # 标量优先 [w,x,y,z]
        return pc, quat_gt

    @staticmethod
    def _shuffle_pc(pc, pc_gt):
        """打乱点云（复用原有逻辑）"""
        order = np.arange(pc.shape[0])
        random.shuffle(order)
        return pc[order], pc_gt[order]

    def _pad_data(self, data, pad_size=None):
        """数据填充（复用原有逻辑）"""
        if pad_size is None:
            pad_size = self.max_num_part
        data = np.array(data)
        pad_shape = (pad_size,) + tuple(data.shape[1:]) if len(data.shape) > 1 else (pad_size,)
        pad_data = np.zeros(pad_shape, dtype=data.dtype)
        pad_data[:data.shape[0]] = data
        return pad_data

    def _sample_fixed_points(self, part_pcs):
        """对每个零件点云采样固定总数的点
        args:
            part_pcs: 零件点云列表 [P1, P2, ..., Pn]
        return:
            total_pcs: sample points [P1, P2, ..., Pn], here len(Pi) is fixed
            nps: 各零件采样点数
        """
        total_pcs = []
        nps = []  # 各零件采样点数
        num_parts = len(part_pcs)
        
        # 按零件数均分点数，最后一个零件补全
        base_n = self.num_points // num_parts
        remain_n = self.num_points % num_parts
        
        for i, pc in enumerate(part_pcs):
            n = base_n + (1 if i == num_parts - 1 else 0) * remain_n
            # 采样/填充到指定点数
            if len(pc) >= n:
                sample_idx = np.random.choice(len(pc), n, replace=False)
                sampled_pc = pc[sample_idx]
            else:
                sample_idx = np.random.choice(len(pc), n, replace=True)
                sampled_pc = pc[sample_idx]
            total_pcs.append(sampled_pc)
            nps.append(n)
        
        return total_pcs, nps

    def __getitem__(self, index):
        """重写样本读取逻辑"""
        # 1. 加载预处理后的点云和元信息
        sample_path = self.data_list[index]
        with open(sample_path, 'rb') as f:
            sample_data = pickle.load(f)
        
        part_pcs = sample_data['part_pcs']
        num_parts = sample_data['num_parts']
        part_rotations = sample_data['part_rotations']
        part_translations = sample_data['part_translations']
        assert len(part_pcs) == len(part_rotations)
        # 2. 对点云采样固定总数的点
        cur_pts, nps = self._sample_fixed_points(part_pcs)
        cur_pts_gt_list = [] # GT点云（原始未变换）
        for i, pc in enumerate(cur_pts):
            cur_pts_gt = pc.copy()
            # 3. 中心化+旋转+洗牌（兼容原有逻辑）
            rc_pts, _ = self._recenter_pc(pc)
            rc_pts = pc
            # cur_pts, gt_quat = self._rotate_pc(cur_pts)
            cur_pts[i], cur_pts_gt = self._shuffle_pc(rc_pts, cur_pts_gt)
            cur_pts_gt = (cur_pts_gt - np.array(part_translations[i])) @ \
                np.array(part_rotations[i])
            cur_pts_gt_list.append(cur_pts_gt)

        cur_pts = np.concatenate(cur_pts, axis=0).astype(np.float32)
        cur_pts_gt = np.concatenate(cur_pts_gt_list, axis=0).astype(np.float32)
        
        # 4. 构造位姿数据（兼容原有格式）
        # 旋转四元数：从旋转矩阵转换（标量优先）
        part_quat = []
        for rot_mat in part_rotations:
            quat = R.from_matrix(rot_mat.T).as_quat()
            part_quat.append(quat[[3, 0, 1, 2]])  # [w,x,y,z]
        # cur_pts_gt = np.linalg.inv(np.stack(part_rotations)) @ \
        #     (cur_pts_gt - np.stack(part_translations))
        # cur_pts_gt = cur_pts_gt.reshape(-1, 3)
        # 5. 数据填充（适配批量训练）
        part_quat = self._pad_data(np.stack(part_quat, axis=0)).astype(np.float32)
        part_trans = self._pad_data(np.stack(part_translations, axis=0)).astype(np.float32)
        n_pcs = self._pad_data(np.array(nps)).astype(np.int64)
        valids = np.zeros(self.max_num_part, dtype=np.float32)
        valids[:num_parts] = 1.0
        
        # 6. 构造返回字典（完全兼容原有格式）
        label_thresholds = np.ones([self.num_points], dtype=np.float32) * self.fracture_label_threshold
        data_dict = {
            "part_pcs": cur_pts.astype(np.float32),  # [N_sum, 3]
            "gt_pcs": cur_pts_gt.astype(np.float32),  # [N_sum, 3]
            "part_valids": valids,  # [max_num_part]
            "part_quat": part_quat,  # [max_num_part, 4]
            "part_trans": part_trans,  # [max_num_part, 3]
            "n_pcs": n_pcs,  # [max_num_part]
            "data_id": index,
            "critical_label_thresholds": label_thresholds,  # [N_sum]
        }
        return data_dict


def build_custom_dataset(cfg):
    """适配新数据集的DataLoader构建函数"""
    data_dict = dict(
        data_dir=cfg.DATA.DATA_DIR.format("train"),  # 预处理后的数据目录
        data_fn=cfg.DATA.DATA_FN.format("train"),  # {train/test}_valid_pcs_list.txt
        num_points=cfg.DATA.NUM_PC_POINTS,
        min_num_part=cfg.DATA.MIN_NUM_PART,
        max_num_part=cfg.DATA.MAX_NUM_PART,
        shuffle_parts=cfg.DATA.SHUFFLE_PARTS,
        rot_range=cfg.DATA.ROT_RANGE,
        overfit=cfg.DATA.OVERFIT,
        length=cfg.DATA.LENGTH * cfg.BATCH_SIZE,
        fracture_label_threshold=cfg.DATA.FRACTURE_LABEL_THRESHOLD,
    )
    train_set = AllPieceMatchingDataset(**data_dict)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    # 验证集配置
    data_dict["data_dir"] = cfg.DATA.DATA_DIR.format("test")
    data_dict["data_fn"] = cfg.DATA.DATA_FN.format("test")  # val_valid_pcs_list.txt
    data_dict["shuffle_parts"] = False
    data_dict["length"] = cfg.DATA.TEST_LENGTH
    val_set = AllPieceMatchingDataset(**data_dict)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )
    return train_loader, val_loader