import torch.nn as nn
import os
import torch
import numpy as np
import pickle
from torch.nn import functional as F
import torchvision.transforms as transforms

try:
    from detr.main import (
        build_ACT_model_and_optimizer,
        build_CNNMLP_model_and_optimizer,
    )
except:
    from .detr.main import (
        build_ACT_model_and_optimizer,
        build_CNNMLP_model_and_optimizer,
    )
import IPython

e = IPython.embed


class ACTPolicy(nn.Module):
    """
    ACT 模型的训练/推理封装。

    初学者可以先把 ACT 理解成：
    - 输入：当前机器人关节状态 qpos + 多相机图像 image
    - 输出：未来一小段动作序列 action chunk

    这里的 ACT 底层是一个 CVAE 风格的 Transformer：
    - 训练时输入真实 actions，让模型学习重建未来动作，并用 KL 约束 latent 分布。
    - 推理时不输入 actions，模型从 prior 采样/生成未来动作序列。
    """

    def __init__(self, args_override, RoboTwin_Config=None):
        super().__init__()
        # build_ACT_model_and_optimizer 会创建真正的 ACT 网络和优化器。
        # 具体模型结构在 policy/ACT/detr/models/detr_vae.py 里。
        model, optimizer = build_ACT_model_and_optimizer(args_override, RoboTwin_Config)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        # KL loss 权重：ACT 是 CVAE，除了动作重建误差，还会约束 latent 分布不要偏离标准正态太远。
        self.kl_weight = args_override["kl_weight"]
        print(f"KL Weight {self.kl_weight}")

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        # 图像在 utils.py 里已经从 [0, 255] 变成 [0, 1]。
        # 这里再用 ImageNet 均值/方差做标准化，方便使用 torchvision backbone。
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            # 训练时 utils.py 返回的 action 可能被 pad 到 max_action_len。
            # ACT 实际只预测 num_queries 长度的 action chunk，所以这里截断到模型需要的长度。
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            # 训练模式：把真实 future actions 也喂给 CVAE encoder。
            # a_hat 是模型预测的动作序列，mu/logvar 是 latent 分布参数。
            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            # 对每个动作维度计算 L1 误差；reduction="none" 是为了后面能用 is_pad mask 掉 padding。
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            # is_pad=True 的位置是补齐出来的无效动作，不应该参与 loss。
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            # ACT 总损失 = 动作重建 L1 + KL 正则项。
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            # 推理模式：没有真实 actions，模型直接根据 qpos/image 预测未来动作 chunk。
            a_hat, _, (_, _) = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        # imitate_episodes.py 会调用这个函数拿到 optimizer。
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    """
    一个更简单的 baseline policy。

    它不是 ACT 的 action chunking + CVAE 结构，只预测单步动作。
    在学习 ACT 时可以先略过，知道它是对照 baseline 即可。
    """

    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None  # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            # CNNMLP 只预测一步动作，所以只取 action 序列里的第 0 步。
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict["mse"] = mse
            loss_dict["loss"] = loss_dict["mse"]
            return loss_dict
        else:  # inference time
            a_hat = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    """
    计算 CVAE latent 分布和标准正态分布之间的 KL divergence。

    直观理解：
    - encoder 根据真实 actions 得到一个 latent 分布 N(mu, sigma)。
    - KL 项鼓励这个分布不要离标准正态 N(0, 1) 太远。
    - 这样推理时即使没有真实 actions，也能从 prior 生成合理动作。
    """
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    # total_kld 是每个样本所有 latent 维度求和后，再对 batch 求平均。
    total_kld = klds.sum(1).mean(0, True)
    # dimension_wise_kld 可以看每个 latent 维度各自贡献多少 KL。
    dimension_wise_kld = klds.mean(0)
    # mean_kld 是所有样本和所有维度上的平均 KL。
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class ACT:
    """
    ACT 的部署/评测封装。

    ACTPolicy 更偏“训练代码”；这个 ACT 类更偏“运行一个训练好的 policy”：
    - 加载 policy checkpoint 和 dataset_stats.pkl
    - 把环境观测 obs 转成模型输入 tensor
    - 调用模型预测 action chunk
    - 按时间步取出当前要执行的动作
    - 把标准化动作还原回机器人动作空间

    如果你在看 RoboTwin 如何 eval ACT，重点看这个类的 get_action()。
    """

    def __init__(self, args_override=None, RoboTwin_Config=None):
        if args_override is None:
            args_override = {
                "kl_weight": 0.1,  # Default value, can be overridden
                "device": "cuda:0",
            }
        # 创建 ACTPolicy，并切到 eval 模式。评测时不会反向传播。
        self.policy = ACTPolicy(args_override, RoboTwin_Config)
        self.device = torch.device(args_override["device"])
        self.policy.to(self.device)
        self.policy.eval()

        # Temporal aggregation settings
        # temporal_agg 是 ACT 里常用的平滑技巧：
        # 每个时间步都预测未来一段 action chunk，多次预测会对同一个未来时刻给出多个动作建议。
        # temporal aggregation 会把这些建议加权平均，减少动作抖动。
        self.temporal_agg = args_override.get("temporal_agg", False)
        # num_queries 也就是 chunk_size：模型一次预测多少步未来动作。
        self.num_queries = args_override["chunk_size"]
        self.state_dim = RoboTwin_Config.action_dim  # Standard joint dimension for bimanual robot
        self.max_timesteps = 3000  # Large enough for deployment

        # Set query frequency based on temporal_agg - matching imitate_episodes.py logic
        # 不使用 temporal_agg 时，每隔 chunk_size 步查询一次模型，然后顺序执行这段 chunk。
        self.query_frequency = self.num_queries
        if self.temporal_agg:
            # 使用 temporal_agg 时，每一步都重新查询模型。
            self.query_frequency = 1
            # Initialize with zeros matching imitate_episodes.py format
            # all_time_actions[t_query, t_target] 存的是：
            # 在 t_query 时刻预测出来、准备给 t_target 时刻使用的动作。
            self.all_time_actions = torch.zeros([
                self.max_timesteps,
                self.max_timesteps + self.num_queries,
                self.state_dim,
            ]).to(self.device)
            print(f"Temporal aggregation enabled with {self.num_queries} queries")

        self.t = 0  # Current timestep

        # Load statistics for normalization
        ckpt_dir = args_override.get("ckpt_dir", "")
        if ckpt_dir:
            # Load dataset stats for normalization
            # dataset_stats.pkl 来自训练数据，里面保存 qpos/action 的 mean/std。
            # 评测时必须使用和训练时相同的归一化参数。
            stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
            if os.path.exists(stats_path):
                with open(stats_path, "rb") as f:
                    self.stats = pickle.load(f)
                print(f"Loaded normalization stats from {stats_path}")
            else:
                print(f"Warning: Could not find stats file at {stats_path}")
                self.stats = None

            # Load policy weights
            # policy_last.ckpt 是训练保存的模型参数。
            ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
            print("current pwd:", os.getcwd())
            if os.path.exists(ckpt_path):
                loading_status = self.policy.load_state_dict(torch.load(ckpt_path))
                print(f"Loaded policy weights from {ckpt_path}")
                print(f"Loading status: {loading_status}")
            else:
                print(f"Warning: Could not find policy checkpoint at {ckpt_path}")
        else:
            self.stats = None

    def pre_process(self, qpos):
        """Normalize input joint positions"""
        # 环境给的是原始 qpos；模型训练时见到的是标准化 qpos，所以推理前也要标准化。
        if self.stats is not None:
            return (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        return qpos

    def post_process(self, action):
        """Denormalize model outputs"""
        # 模型输出的是标准化 action；真正给机器人执行前要变回原始动作尺度。
        if self.stats is not None:
            return action * self.stats["action_std"] + self.stats["action_mean"]
        return action

    def get_action(self, obs=None):
        """
        根据当前环境观测 obs，返回这一时刻要执行的动作。

        obs 需要包含：
        - "qpos": 当前机器人状态
        - "head_cam"、"left_cam"、"right_cam": 三路相机图像

        返回值 action 是反归一化后的机器人动作，后续会交给环境 take_action() 执行。
        """
        if obs is None:
            return None

        # Convert observations to tensors and normalize qpos - matching imitate_episodes.py
        qpos_numpy = np.array(obs["qpos"])
        qpos_normalized = self.pre_process(qpos_numpy)
        # 增加 batch 维度：从 [state_dim] 变成 [1, state_dim]。
        qpos = torch.from_numpy(qpos_normalized).float().to(self.device).unsqueeze(0)

        # Prepare images following imitate_episodes.py pattern
        # Stack images from all cameras
        curr_images = []
        camera_names = ["head_cam", "left_cam", "right_cam"]
        for cam_name in camera_names:
            curr_images.append(obs[cam_name])
        # 图像堆叠后形状通常是 [num_cameras, C, H, W] 或 [num_cameras, H, W, C]，
        # 具体取决于 deploy_policy 传入 obs 时的格式；这里保持和 imitate_episodes.py 一致。
        curr_image = np.stack(curr_images, axis=0)
        # 增加 batch 维度，变成 [1, num_cameras, ...]。
        curr_image = torch.from_numpy(curr_image).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            # Only query the policy at specified intervals - exactly like imitate_episodes.py
            if self.t % self.query_frequency == 0:
                # all_actions 的形状通常是 [1, num_queries, action_dim]。
                self.all_actions = self.policy(qpos, curr_image)

            if self.temporal_agg:
                # Match temporal aggregation exactly from imitate_episodes.py
                # 把当前时刻预测的一整段未来动作写入时间表。
                self.all_time_actions[[self.t], self.t:self.t + self.num_queries] = (self.all_actions)
                actions_for_curr_step = self.all_time_actions[:, self.t]
                # 取出所有曾经为当前时刻 t 预测过的动作，过滤掉还没填过的全 0 行。
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]

                # Use same weighting factor as in imitate_episodes.py
                # 越新的预测权重越大，越旧的预测权重越小。
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = (torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1))

                # 对多个 action 建议做加权平均，得到当前真正执行的动作。
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                # Direct action selection, same as imitate_episodes.py
                # 不做 temporal aggregation 时，就按 chunk 内部的时间索引顺序取动作。
                raw_action = self.all_actions[:, self.t % self.query_frequency]

        # Denormalize action
        raw_action = raw_action.cpu().numpy()
        action = self.post_process(raw_action)

        # 时间步加一，下一次调用会取下一个动作。
        self.t += 1
        return action
