import os
import random
import numpy as np
import open3d as o3d
import argparse
from dataset.custom_dataset import AllPieceMatchingDataset  # 导入改造后的数据集类

def setup_seed(seed):
    """设置随机种子，保证结果可复现"""
    random.seed(seed)
    np.random.seed(seed)

def visualize_point_cloud(pc_data, window_title="Part PCS Visualization"):
    """
    可视化单样本点云
    Args:
        pc_data: numpy数组，形状[N, 3]，点云数据
        window_title: 可视化窗口标题
    """
    # 1. 转换为Open3D点云格式
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_data.astype(np.float64))  # Open3D要求float64
    
    # 2. 为点云添加随机颜色（增强可视化效果）
    colors = np.random.uniform(0, 1, size=(len(pc_data), 3))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 3. 创建可视化窗口并添加坐标系
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_title, width=800, height=600)
    vis.add_geometry(pcd)
    
    # 添加坐标系（原点，轴长0.1）
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    
    # 4. 交互控制：按Q退出，按SPACE切换随机颜色
    def exit_callback(vis):
        vis.close()
        return False
    
    def change_color_callback(vis):
        new_colors = np.random.uniform(0, 1, size=(len(pc_data), 3))
        pcd.colors = o3d.utility.Vector3dVector(new_colors)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        return False
    
    vis.register_key_callback(ord('Q'), exit_callback)
    vis.register_key_callback(ord(' '), change_color_callback)
    
    # 5. 运行可视化
    vis.run()
    vis.destroy_window()

def main(args=None, cfg=None):
    """主函数：初始化数据集并可视化随机样本"""
    # 设置随机种子
    assert args is not None or cfg is not None
    if args is not None:
        setup_seed(args.seed)
        
        # 1. 初始化改造后的AllPieceMatchingDataset
        dataset = AllPieceMatchingDataset(
            data_dir=args.data_dir,               # 预处理后的数据目录
            data_fn=args.data_fn,                 # 有效样本列表文件（如train_valid_pcs_list.txt）
            num_points=args.num_points,           # 总采样点数
            min_num_part=args.min_num_part,       # 最小零件数
            max_num_part=args.max_num_part,       # 最大零件数
            shuffle_parts=args.shuffle_parts,     # 是否打乱零件顺序
            rot_range=args.rot_range,             # 旋转范围（-1为随机旋转）
            fracture_label_threshold=args.label_threshold  # 兼容参数
        )
    elif cfg is not None:
        # dataset = AllPieceMatchingDataset(
        #     data_dir=cfg.data_dir,               # 预处理后的数据目录
        #     data_fn=cfg.data_fn,                 # 有效样本列表文件（如train_valid_pcs_list.txt）
        #     num_points=cfg.num_points,           # 总采样点数
        #     min_num_part=cfg.min_num_part,       # 最小零件数
        #     max_num_part=cfg.max_num_part,       # 最大零件数
        #     shuffle_parts=cfg.shuffle_parts,     # 是否打乱零件顺序
        #     fracture_label_threshold=cfg.label_threshold  # 兼容参数
        # )
        from dataset import build_custom_dataset
        tra_da, val_da = build_custom_dataset(cfg)
        dataset = tra_da.dataset
    else:
        raise ValueError("cfg or args must be not None")
    
    print(f"数据集总样本数：{len(dataset)}")
    
    # 2. 遍历/随机选择样本可视化
    while True:
        # 随机选择样本索引
        sample_idx = random.randint(0, len(dataset)-1)
        print(f"\n当前可视化样本索引: {sample_idx}")
        
        # 加载样本（__getitem__返回字典）
        sample = dataset[sample_idx]
        part_pcs = sample["gt_pcs"]  # 核心点云数据，形状[N, 3]
        npcs = sample["n_pcs"]
        print(f"有效零件数：{npcs}")
        print(f"rotations info: {sample['part_trans']}")
        # 打印点云基本信息（校验数据准确性）
        print(f"点云形状：{part_pcs.shape}")
        print(f"点云坐标范围: x in [{part_pcs[:,0].min():.4f}, {part_pcs[:,0].max():.4f}], "
              f"y in [{part_pcs[:,1].min():.4f}, {part_pcs[:,1].max():.4f}], "
              f"z in [{part_pcs[:,2].min():.4f}, {part_pcs[:,2].max():.4f}]")
        
        # 3. 可视化点云
        visualize_point_cloud(part_pcs, window_title=f"Sample {sample_idx} - Path {dataset.data_list[sample_idx]}")
        
        # 4. 交互选择是否继续
        user_input = input("是否继续可视化下一个样本？(y/n): ").strip().lower()
        if user_input != 'y':
            print("可视化结束！")
            break

if __name__ == "__main__":
    # 命令行参数配置（方便灵活调整）
    # parser = argparse.ArgumentParser(description="AllPieceMatchingDataset 点云可视化Demo")
    # parser.add_argument("--data_dir", type=str, required=True, 
    #                     help="预处理后的数据集根目录（如/path/to/preprocessed/custom_dataset）")
    # parser.add_argument("--data_fn", type=str, default="train_valid_pcs_list.txt",
    #                     help="有效样本列表文件（如train_valid_pcs_list.txt/val_valid_pcs_list.txt）")
    # parser.add_argument("--num_points", type=int, default=1000,
    #                     help="每个样本的总采样点数（需与数据集初始化参数一致）")
    # parser.add_argument("--min_num_part", type=int, default=2, help="最小零件数")
    # parser.add_argument("--max_num_part", type=int, default=20, help="最大零件数")
    # parser.add_argument("--shuffle_parts", action="store_true", default=False, help="是否打乱零件顺序")
    # parser.add_argument("--rot_range", type=float, default=-1, help="旋转角度范围（-1为随机旋转）")
    # parser.add_argument("--label_threshold", type=float, default=0.025, help="断裂标签阈值")
    # parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # args = parser.parse_args()
    from utils.config import cfg_from_file
    cfg_from_file('experiments/custom.yaml')
    from utils.config import cfg
    # 检查Open3D版本（确保兼容性）
    assert o3d.__version__ >= "0.16.0", "请升级Open3D至0.16.0及以上版本"
    
    # 启动可视化
    main(cfg=cfg)