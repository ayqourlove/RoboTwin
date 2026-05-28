# RoboTwin 中的 ACT 入门说明

这份说明面向“有深度学习基础，但刚开始接触 VLA / 机器人模仿学习”的读者，目标是回答四个问题：

1. `policy/ACT` 里的 ACT 到底在做什么
2. 它的输入和输出分别是什么
3. 它在 RoboTwin 里是怎么训练的
4. RoboTwin 的训练数据是怎么采集出来的

本文基于当前仓库代码，以及 RoboTwin 官方文档、ACT 原始项目资料整理而成。为了方便你对照阅读，我会尽量把概念和代码文件对应起来。

## 1. 先说结论：RoboTwin 里的 ACT 不是“完整语言版 VLA”

RoboTwin 是一个 VLA benchmark，但 `policy/ACT` 这个 baseline 本身并不接收文本语言输入。  
从代码看，ACT 的运行时输入只有：

- 3 路 RGB 图像
- 当前机器人关节状态 `qpos`

输出是：

- 一串未来动作 `action chunk`

也就是说，这里的 ACT 更准确地说是一个“视觉 + 本体状态 -> 动作序列”的 imitation learning policy，而不是像 OpenVLA / Pi0 那样显式接收文本 instruction 的 VLA 大模型。

这点很重要，因为初学者很容易把“benchmark 支持语言任务”与“某个具体 baseline 的模型输入一定包含语言”混为一谈。

## 2. ACT 的核心思想：一次不要只预测一步，而是预测一小段动作

ACT 全名是 `Action Chunking with Transformers`。它最核心的想法是：

- 普通行为克隆常常每一步只预测一个动作
- 但精细双臂操作里，单步预测很容易抖、停顿、误差积累
- 所以 ACT 让模型一次预测未来一段动作，也就是一个 `chunk`

在 RoboTwin 这份实现里，这个 chunk 长度由 `chunk_size` 控制，训练脚本默认是 `50`。

所以你可以把 ACT 粗略理解成：

> 给模型当前时刻的观测，让它直接规划“接下来 50 步我应该怎么动”

这也是为什么它在代码里不是输出一个动作，而是输出一个形状近似为：

```python
[batch_size, chunk_size, action_dim]
```

的动作序列。

## 3. 先看整体流程：从 RoboTwin 采数据到 ACT 训练

在这个仓库里，ACT 的完整流程可以概括为 3 段：

### 3.1 RoboTwin 自动生成专家轨迹

入口：

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
```

对应代码：

- `collect_data.sh`
- `script/collect_data.py`
- `envs/_base_task.py`
- 各任务文件，例如 `envs/beat_block_hammer.py`

### 3.2 把 RoboTwin 原始 HDF5 转成 ACT 训练格式

入口：

```bash
cd policy/ACT
bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
```

对应代码：

- `policy/ACT/process_data.sh`
- `policy/ACT/process_data.py`

### 3.3 训练 ACT

入口：

```bash
cd policy/ACT
bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${gpu_id}
```

对应代码：

- `policy/ACT/train.sh`
- `policy/ACT/imitate_episodes.py`
- `policy/ACT/act_policy.py`
- `policy/ACT/detr/models/detr_vae.py`

## 4. RoboTwin 是怎么采训练数据的

这是你理解整个 benchmark 的关键，因为 RoboTwin 的数据不是“人手 teleop 录一遍”那种模式，而是“任务脚本 + 规划器 + 自动回放录制”的模式。

### 4.1 任务本身由 `envs/*.py` 里的专家脚本定义

每个任务通常都有两个关键函数：

- `setup_demo(...)`
- `play_once(...)`

例如 `envs/beat_block_hammer.py` 里：

- `setup_demo` 会初始化场景
- `play_once` 会调用一系列机器人运动原语，比如抓取、抬起、放置

所以 RoboTwin 的“专家演示”本质上是任务脚本生成的成功轨迹，而不是本仓库里的人工 teleoperation 录制。

### 4.2 采集分成两个阶段：先找成功种子，再正式录制

`script/collect_data.py` 的逻辑不是上来就录视频和 HDF5，而是先做一次“成功轨迹搜索”：

#### 阶段 A：搜索成功随机种子

代码里大致是：

1. 调 `TASK_ENV.setup_demo(..., seed=epid, need_plan=True)`
2. 调 `TASK_ENV.play_once()`
3. 检查 `plan_success` 和 `check_success()`
4. 如果成功，就把这个 `seed` 记到 `seed.txt`
5. 同时把规划出来的关节路径保存到 `_traj_data/episode{i}.pkl`

这里的 `_traj_data` 不是训练数据本身，而是“之后重放用的轨迹缓存”。

#### 阶段 B：重放成功种子并正式录制

找到足够多的成功种子后，代码会：

1. 重新根据 `seed.txt` 重建场景
2. 读回 `_traj_data` 中保存的 `left_joint_path` / `right_joint_path`
3. 执行一次无渲染的数据录制
4. 每一步把观测和动作先存成 `.pkl`
5. 最后把整段 `.pkl` 合并成一个 `episode{i}.hdf5` 和一个视频

所以你可以把 RoboTwin 的数据采集理解成：

> 先确认“这个随机场景能成功做出来”，再在同一个场景上稳定回放并录制成训练样本

这样做的好处是：

- 不会把大量失败样本直接录进训练集
- 每条轨迹都来自一个成功执行的专家过程
- 可以很好地配合大规模 domain randomization

### 4.3 原始 RoboTwin 数据里录了什么

`task_config/demo_randomized.yml` 里可以看到，默认会打开：

- `rgb: true`
- `endpose: true`
- `qpos: true`

在 `envs/_base_task.py` 的 `get_obs()` 里，单步观测会被整理成一个字典，核心字段包括：

- `observation/<camera_name>/rgb`
- `joint_action/left_arm`
- `joint_action/left_gripper`
- `joint_action/right_arm`
- `joint_action/right_gripper`
- `joint_action/vector`
- `endpose/...`

随后 `envs/utils/pkl2hdf5.py` 会把这一串逐帧 `.pkl` 合并成一个 episode 级别的 HDF5。

### 4.4 原始数据放在哪里

一次采集结束后，典型目录会长这样：

```text
data/${task_name}/${task_config}/
  data/episode0.hdf5
  data/episode1.hdf5
  video/episode0.mp4
  instructions/
  _traj_data/
  scene_info.json
  seed.txt
```

其中：

- `data/episode{i}.hdf5` 是原始训练数据
- `video/` 方便肉眼检查轨迹质量
- `instructions/` 是语言描述
- `_traj_data/` 和 `seed.txt` 是自动生成过程的辅助缓存

## 5. RoboTwin 采集时做了哪些随机化

RoboTwin 2.0 很强调 domain randomization。当前仓库和官方文档里，主要随机化包括：

- cluttered table：桌面杂物
- random background：背景纹理
- random light：光照颜色/强度
- random table height：桌面高度
- random head camera distance：头部相机扰动

这也是为什么官方建议尽量自己采数据，而不是只用现成数据集：  
同一个任务、不同 embodiment、不同相机设置、不同随机化强度，得到的数据分布可能差很多。

## 6. ACT 训练前的数据预处理：RoboTwin 原始格式 -> ACT 格式

ACT 并不直接读取 RoboTwin 原始 `data/episode{i}.hdf5`，而是先做一次格式转换。

入口脚本：

```bash
cd policy/ACT
bash process_data.sh beat_block_hammer demo_randomized 50
```

### 6.1 `process_data.py` 做了什么

它会读取原始数据里的：

- `joint_action/left_arm`
- `joint_action/left_gripper`
- `joint_action/right_arm`
- `joint_action/right_gripper`
- 3 路相机图像

然后生成 ACT 训练所需的统一格式：

- `action`
- `observations/qpos`
- `observations/images/cam_high`
- `observations/images/cam_right_wrist`
- `observations/images/cam_left_wrist`

最终输出目录类似：

```text
policy/ACT/processed_data/sim-${task_name}/${task_config}-${expert_data_num}/
  episode_0.hdf5
  episode_1.hdf5
  ...
```

### 6.2 这里的 `qpos` 和 `action` 分别是什么

在当前实现里：

- `qpos[t] = [left_arm, left_gripper, right_arm, right_gripper]`
- `action[t]` 也是同样 14 维结构

也就是说，ACT 学的是 joint-position 控制命令。

不过要注意，`process_data.py` 里的时间对齐不是一个非常“教科书式”的写法，而是一个近似实现：

- 第 `j` 帧状态会被写入 `qpos`
- 从 `j=1` 开始，当前状态会被追加到 `actions`
- 最后一轮循环里会再次复用上一帧的 `state`，因此末尾会出现一次重复的最后有效动作

所以更准确地说，这份数据近似在学：

> 用当前观测预测接下来要执行的 joint-position 目标

对入门学习来说，你先把它理解成“当前观测 -> 未来 joint 目标序列”就足够了；等你之后要自己改数据管线时，再回来处理这个时间对齐细节。

### 6.3 一个重要限制：这份 ACT 实现基本假设 `action_dim = 14`

虽然 RoboTwin 官方文档说 action 维度会随 embodiment 改变，但当前 `policy/ACT` 的训练代码里多处直接写死了 `14`：

- `policy/ACT/imitate_episodes.py`
- `policy/ACT/detr/models/detr_vae.py`
- `policy/ACT/deploy_policy.yml`

所以当前这份 ACT baseline 本质上是为 `aloha-agilex` 这一类 14 维双臂 joint 控制写的。  
如果你换成别的 embodiment，理论上 RoboTwin 环境可以适配，但这份 ACT 训练代码未必能直接用。

这对初学者很值得注意，因为它说明：

> benchmark 的“跨 embodiment 能力”与某个 baseline 实现是否真的做成“可变 action_dim”是两回事

## 7. ACT 的输入、输出和内部结构

### 7.1 训练时 ACT 的单个样本长什么样

`policy/ACT/utils.py` 中的 `EpisodicDataset` 会这样采样：

1. 随机挑一个 episode
2. 在这个 episode 内随机挑一个起点 `start_ts`
3. 取这一时刻的观测：
   - `qpos[start_ts]`
   - 3 路图像 `images[cam][start_ts]`
4. 取从该时刻往后的整段动作序列
5. 把动作序列 pad 到数据集中的最大长度，并生成 `is_pad`

所以训练时每个样本并不是“整段轨迹一起喂进去”，而是：

- 一个当前观测
- 对应的一段未来动作

### 7.2 模型输入是什么

训练和推理时，核心输入都是：

- `qpos`：14 维当前关节状态
- `image`：3 路 RGB 图像

在当前默认配置下，相机名是：

- `cam_high`
- `cam_right_wrist`
- `cam_left_wrist`

图像会先缩放到 `640x480`，再转成 `[num_cam, 3, H, W]` 格式，并按 ImageNet 均值方差归一化。

### 7.3 模型输出是什么

ACT 的主输出是：

- `a_hat`: 未来 `chunk_size` 步动作

在默认训练脚本中：

- `chunk_size = 50`
- `action_dim = 14`

所以主输出可以理解为：

```python
[50, 14]
```

另外模型还会输出：

- `mu, logvar`：CVAE latent 的均值和方差
- `is_pad_hat`：padding 相关预测

但注意：当前 RoboTwin 这份封装里，训练损失真正用到的是：

- `L1(action, a_hat)`
- `KL(mu, logvar)`

`is_pad_hat` 在这里并没有单独参与 loss。

### 7.4 为什么它能“一次预测一串动作”

`policy/ACT/detr/models/detr_vae.py` 里，ACT 的结构可以概括成三部分：

#### 部分 A：图像编码

- 每个相机图像先过 ResNet18 backbone
- 提取特征后映射到 transformer hidden dim
- 多相机特征在宽度维拼接

一个细节是：当前代码对三路相机使用的是共享 backbone 权重。

#### 部分 B：动作序列编码器（CVAE encoder）

训练时会把：

- 当前 `qpos`
- 真实未来动作序列 `actions`

一起送进 encoder，得到 latent 分布 `mu, logvar`。  
这相当于告诉模型：在训练阶段，除了看当前观测，还可以参考“这段未来动作长什么样”，从而学到一个更稳定的动作分布表示。

推理时没有真实未来动作，所以直接用零向量作为 prior latent。

#### 部分 C：Transformer decoder + action queries

模型里有 `num_queries = chunk_size` 个 query embedding。  
你可以把每个 query 想成“我要预测未来第 k 个动作”的槽位。

所以 decoder 一次前向就会同时吐出：

- 第 1 步动作
- 第 2 步动作
- ...
- 第 `chunk_size` 步动作

这就是 action chunking。

## 8. ACT 的损失函数和训练方式

`policy/ACT/act_policy.py` 里，训练损失非常直接：

```python
loss = L1(actions, a_hat) + kl_weight * KL(mu, logvar)
```

其中：

- `L1` 负责让预测动作接近专家动作
- `KL` 负责约束 CVAE latent，不要过拟合成完全记忆训练集

当前默认超参数来自 `policy/ACT/train.sh`：

- `kl_weight = 10`
- `chunk_size = 50`
- `hidden_dim = 512`
- `dim_feedforward = 3200`
- `batch_size = 8`
- `num_epochs = 6000`
- `lr = 1e-5`

训练流程在 `policy/ACT/imitate_episodes.py` 中：

1. 从处理后的数据集加载样本
2. 计算全数据集的 `qpos_mean/std` 和 `action_mean/std`
3. 保存到 `dataset_stats.pkl`
4. 按 `80% / 20%` 划分 train / val
5. 每个 epoch 先做 validation，再做 training
6. 保存 `policy_best.ckpt` 和 `policy_last.ckpt`

## 9. 推理时 ACT 怎么出动作

运行时入口可以看：

- `policy/ACT/act_policy.py`
- `policy/ACT/deploy_policy.py`

推理时大致是：

1. 读取当前观测
2. 把 `qpos` 做标准化
3. 把 3 路图像堆叠后送进 policy
4. 模型输出一个动作 chunk
5. 再把动作反标准化
6. 从 chunk 里取当前该执行的那一步动作

### 9.1 不开 temporal aggregation

默认情况下：

- 每隔 `chunk_size` 步重新 query 一次模型
- 中间直接顺序执行这一整段 chunk

也就是：

> 第 0 步预测未来 50 步，然后按顺序执行  
> 第 50 步再重新预测下一段 50 步

### 9.2 开 temporal aggregation

如果开启 `temporal_agg`：

- 每一步都重新预测一个 chunk
- 对“所有覆盖当前时刻的预测”做指数加权平均

这样做通常更平滑，但计算更贵。

你可以把它理解成：

> 每一时刻都听取多个“过去时刻对当前动作的建议”，再做一个加权融合

## 10. 用最小心智模型理解 ACT

如果你现在刚入门，可以先不要陷进所有 transformer 细节，先记住下面这个最小模型：

- RoboTwin 自动生成成功的专家 joint 轨迹
- `process_data.py` 把原始轨迹整理成 `当前观测 -> 未来动作序列`
- ACT 学的是：看到当前图像和关节状态后，直接预测未来一小段 joint-position 动作
- 一次预测一串动作，比一步一步贪心预测更稳定
- 训练时用 `L1 + KL`
- 推理时可选 temporal aggregation 来减少抖动

## 11. 建议你的阅读顺序

如果你准备真正把这套代码读明白，我建议按这个顺序：

1. `task_config/demo_randomized.yml`  
   先知道 RoboTwin 采什么数据、开了哪些随机化。
2. `script/collect_data.py`  
   先看“种子搜索 + 重放录制”的总流程。
3. `envs/_base_task.py`  
   重点看 `get_obs()`、`_take_picture()`、`merge_pkl_to_hdf5_video()`。
4. 任意一个任务文件，例如 `envs/beat_block_hammer.py`  
   看专家轨迹究竟是怎么写出来的。
5. `policy/ACT/process_data.py`  
   理解原始 RoboTwin 数据如何变成 ACT 的训练输入。
6. `policy/ACT/utils.py`  
   看 dataloader 如何从 episode 中随机截取训练样本。
7. `policy/ACT/act_policy.py`  
   先把输入输出和 loss 看懂。
8. `policy/ACT/detr/models/detr_vae.py`  
   最后再看模型细节。

## 12. 推荐你先跑的最小实验

如果你想把“读代码”变成“边跑边理解”，建议先选一个简单任务，比如：

```bash
bash collect_data.sh beat_block_hammer demo_clean 0
cd policy/ACT
bash process_data.sh beat_block_hammer demo_clean 50
bash train.sh beat_block_hammer demo_clean 50 0 0
```

然后重点检查三件事：

1. 原始数据目录下的 `video/episode0.mp4`  
   确认专家轨迹是不是稳定、成功。
2. `policy/ACT/processed_data/.../episode_0.hdf5`  
   确认 `qpos`、`action`、三路图像格式是否理解正确。
3. `policy/ACT/act_ckpt/.../dataset_stats.pkl` 和 loss 曲线  
   确认标准化和训练过程是否正常。

## 13. 你现在最该记住的几个点

- RoboTwin 的 ACT baseline 不是显式语言模型
- 它主要学习 `视觉 + qpos -> action chunk`
- 训练数据来自 RoboTwin 自动生成的成功专家轨迹
- 数据采集流程是“先找成功 seed，再回放录制”
- 当前实现强依赖 14 维 joint-position 控制
- 如果你先把数据流和输入输出搞清楚，再看 transformer 细节，会轻松很多

## 14. 参考资料

### RoboTwin 官方资料

- RoboTwin README: https://github.com/RoboTwin-Platform/RoboTwin
- RoboTwin ACT 文档: https://robotwin-platform.github.io/doc/usage/ACT.html
- RoboTwin Collect Data 文档: https://robotwin-platform.github.io/doc/usage/collect-data.html
- RoboTwin Configurations 文档: https://robotwin-platform.github.io/doc/usage/configurations.html
- RoboTwin Domain Randomization 文档: https://robotwin-platform.github.io/doc/usage/domain-randomization.html
- RoboTwin Control Robot 文档: https://robotwin-platform.github.io/doc/usage/control-robot.html

### ACT 原始资料

- ACT GitHub: https://github.com/tonyzhaozh/act
- ALOHA / ACT 作者主页索引: https://tonyzhaozh.github.io/
- 论文标题：Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware

### 当前仓库中最关键的代码文件

- `collect_data.sh`
- `script/collect_data.py`
- `task_config/demo_randomized.yml`
- `envs/_base_task.py`
- `envs/beat_block_hammer.py`
- `policy/ACT/process_data.py`
- `policy/ACT/utils.py`
- `policy/ACT/imitate_episodes.py`
- `policy/ACT/act_policy.py`
- `policy/ACT/detr/models/detr_vae.py`
