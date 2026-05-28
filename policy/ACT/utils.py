import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython

e = IPython.embed


class EpisodicDataset(torch.utils.data.Dataset):
    """
    ACT 训练用的数据集类。

    一个 RoboTwin/ACT 数据集目录里通常有很多 episode_0.hdf5、episode_1.hdf5 ...
    每个 episode 是一条专家演示轨迹。这个 Dataset 每次 __getitem__ 不返回完整 episode，
    而是随机抽一个时间点 start_ts，返回：
    - 当前时刻的多相机图像 image_data
    - 当前时刻的机器人状态 qpos_data
    - 从当前时刻开始的一整段未来动作 action_data
    - 哪些 action 是补齐出来的 is_pad

    这样 ACT 学到的是：“看到当前观测后，预测接下来一段动作序列”。
    """

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, max_action_len):
        super(EpisodicDataset).__init__()
        # episode_ids 是这个 Dataset 负责读取哪些 episode，例如 [0, 3, 7]。
        self.episode_ids = episode_ids
        # dataset_dir 指向 process_data.py 生成的 processed_data 目录。
        self.dataset_dir = dataset_dir
        # camera_names 决定读取哪些相机，例如 cam_high、cam_right_wrist、cam_left_wrist。
        self.camera_names = camera_names
        # norm_stats 保存 qpos/action 的均值和标准差，用来做标准化。
        self.norm_stats = norm_stats
        # 不同 episode 长度可能不同，训练 batch 需要统一长度，所以动作序列会 pad 到这个长度。
        self.max_action_len = max_action_len
        self.is_sim = None
        # 读取一次样本以初始化 self.is_sim；当前 RoboTwin 分支里 is_sim 实际会保持 None。
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        # Dataset 的长度不是帧数，而是 episode 数。
        # 每次访问某个 episode 时，内部再随机抽一个时间点。
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False

        # 找到这次要读取的 episode 文件。
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            is_sim = None
            # /action 是 HDF5 根目录下的 dataset，shape 通常是 [T, action_dim]。
            original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                # 训练时随机抽一个时间点，让同一条 episode 可以产生很多不同训练样本。
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            # qpos 是当前时刻机器人状态，shape 通常是 [action_dim]。
            qpos = root["/observations/qpos"][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                # 每个相机只取 start_ts 这一帧图像。
                # 图像在 process_data.py 中已经保存成 uint8 数组，shape 是 [H, W, C]。
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root["/action"][start_ts:]
                action_len = episode_len - start_ts
            else:
                # 这里向前取一帧是原 ACT 代码里的对齐技巧：
                # 希望“当前观测”和“下一步动作”在时间上更接近。
                action = root["/action"][max(0, start_ts - 1):]  # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned

        self.is_sim = is_sim
        # 未来动作长度会随着 start_ts 变化：
        # 如果从 episode 后半段开始，剩余动作更短；为了组成 batch，需要补齐到 max_action_len。
        padded_action = np.zeros((self.max_action_len, action.shape[1]), dtype=np.float32)  # 根据max_action_len初始化
        padded_action[:action_len] = action
        # is_pad 标记哪些位置是 padding。False 表示真实动作，True 表示补齐出来的无效动作。
        # 训练 loss 会用它忽略 padding 部分。
        is_pad = np.ones(self.max_action_len, dtype=bool)  # 初始化为全1（True）
        is_pad[:action_len] = 0  # 前action_len个位置设置为0（False），表示非填充部分

        # new axis for different cameras
        # 把多路相机堆在第 0 维：
        # [num_cameras, H, W, C]，例如 [3, 480, 640, 3]。
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        # PyTorch/CNN 通常要求图像是 channel-first：
        # 从 [camera, H, W, C] 转成 [camera, C, H, W]。
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # normalize image and change dtype to float
        # 图像从 uint8 [0, 255] 转为 float [0, 1]。
        image_data = image_data / 255.0
        # qpos/action 标准化后，模型更容易训练：
        # 标准化公式是 (x - mean) / std。
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        # DataLoader 会把多个样本自动叠成 batch：
        # image_data -> [B, num_cameras, C, H, W]
        # qpos_data -> [B, action_dim]
        # action_data -> [B, max_action_len, action_dim]
        # is_pad -> [B, max_action_len]
        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    """
    统计整个数据集里的 qpos/action 均值和标准差。

    初学者要记住：模型训练前通常会对连续数值做标准化，
    让每一维大致变成均值 0、方差 1。这样不同量纲的关节角/夹爪值不会让优化变得困难。
    """
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            # [()] 表示把整个 HDF5 dataset 一次性读成 numpy 数组。
            qpos = root["/observations/qpos"][()]  # Assuming this is a numpy array
            action = root["/action"][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    # Pad all tensors to the maximum size
    # 每条 episode 的 T 可能不同，直接 torch.stack 会失败。
    # 所以先找到最长长度，再把短 episode 用最后一帧补齐。
    max_qpos_len = max(q.size(0) for q in all_qpos_data)
    max_action_len = max(a.size(0) for a in all_action_data)

    padded_qpos = []
    for qpos in all_qpos_data:
        current_len = qpos.size(0)
        if current_len < max_qpos_len:
            # Pad with the last element
            # 用最后一帧补齐比用 0 更自然，因为机器人会保持在最后状态。
            pad = qpos[-1:].repeat(max_qpos_len - current_len, 1)
            qpos = torch.cat([qpos, pad], dim=0)
        padded_qpos.append(qpos)

    padded_action = []
    for action in all_action_data:
        current_len = action.size(0)
        if current_len < max_action_len:
            pad = action[-1:].repeat(max_action_len - current_len, 1)
            action = torch.cat([action, pad], dim=0)
        padded_action.append(action)

    all_qpos_data = torch.stack(padded_qpos)
    all_action_data = torch.stack(padded_action)
    all_action_data = all_action_data

    # normalize action data
    # dim=[0, 1] 表示同时跨 episode 维和时间维求统计量，保留 action_dim 这一维。
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    # 防止某些维度标准差太小，除以 std 时数值爆掉。
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {
        # squeeze 后形状通常是 [action_dim]，方便 __getitem__ 中和单个样本广播相减。
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos,
    }

    return stats, max_action_len


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    """
    构建训练/验证 DataLoader。

    这个函数是训练脚本进入数据处理的主要入口：
    1. 按 episode 划分 train/val。
    2. 计算 qpos/action 的归一化统计量。
    3. 创建 EpisodicDataset。
    4. 用 PyTorch DataLoader 产出 batch。
    """
    print(f"\nData from: {dataset_dir}\n")
    # obtain train test split
    train_ratio = 0.8
    # 这里是按 episode 划分，而不是按帧划分。
    # 这样验证集不会和训练集共享同一条专家轨迹。
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats, max_action_len = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, max_action_len)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, max_action_len)
    # DataLoader 负责把 Dataset 返回的单样本合并成 batch，并在训练时打乱样本顺序。
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils


def sample_box_pose():
    # 下面两个 sample_* 函数来自原 ACT 示例环境，用于随机生成简单物体位姿。
    # 在 RoboTwin 的 ACT 数据处理主链路中通常不会用到。
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions


def compute_dict_mean(epoch_dicts):
    # 把一个 epoch 内多个 step 记录的 loss 字典求平均，常用于日志输出。
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    # 从计算图中分离 tensor，避免日志统计时继续保留反向传播图。
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    # 固定随机种子，方便复现实验结果。
    torch.manual_seed(seed)
    np.random.seed(seed)
