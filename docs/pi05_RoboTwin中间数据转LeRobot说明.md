# pi05：将 RoboTwin 中间数据转换为 LeRobot 数据集

本文说明下面这个脚本的用途、输入输出和内部转换逻辑：

```text
policy/pi05/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py
```

它是 RoboTwin 数据进入 pi05 训练流程的第二个转换阶段。第一个阶段由
`policy/pi05/scripts/process_data.py` 完成。

## 1. 整体数据链路

```text
RoboTwin 采集结果
data/<task>/<setting>/data/episode0.hdf5
data/<task>/<setting>/instructions/episode0.json
        │
        │ scripts/process_data.py
        ▼
ALOHA 风格中间数据
policy/pi05/processed_data/<task>-<setting>-<num>/episode_0/
├── episode_0.hdf5
└── instructions.json
        │
        │ convert_aloha_data_to_lerobot_robotwin.py
        ▼
LeRobot 数据集
~/.cache/huggingface/lerobot/<repo_id>/
├── data/
└── meta/
        │
        ├── compute_norm_stats.py
        └── train.py
```

第二阶段不会重新计算机器人动作，也不会再次移动 state/action 的时间位置。它主要做三件事：

1. 声明 LeRobot 数据集每一帧有哪些 feature。
2. 读取中间 HDF5，把状态、动作和三路相机图像逐帧加入 LeRobot。
3. 为每条 episode 选择一条语言指令，生成 Parquet 数据和 LeRobot 元数据。

## 2. 针对当前数据的运行命令

先在 `policy/pi05` 目录完成第一阶段：

```bash
cd /home/gzy/vla/RoboTwin/policy/pi05
bash process_data_pi05.sh adjust_bottle_singlearm demo_clean 100
```

预期产生：

```text
processed_data/adjust_bottle_singlearm-demo_clean-100/
```

然后运行第二阶段：

```bash
bash generate.sh \
  processed_data/adjust_bottle_singlearm-demo_clean-100 \
  adjust_bottle_singlearm
```

`generate.sh` 实际执行的是：

```bash
uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
  --raw-dir processed_data/adjust_bottle_singlearm-demo_clean-100 \
  --repo-id adjust_bottle_singlearm
```

默认输出位置为：

```text
~/.cache/huggingface/lerobot/adjust_bottle_singlearm/
```

这里的 `repo_id` 不只是输出文件夹名。后续还必须把训练配置中的
`LeRobotAlohaDataConfig(repo_id=...)` 改成完全相同的值。

## 3. 输入数据格式

脚本会递归扫描 `--raw-dir`，收集所有扩展名为 `.hdf5` 的文件。每个 HDF5 旁边必须存在
一个 `instructions.json`：

```text
episode_0/
├── episode_0.hdf5
└── instructions.json
```

HDF5 的必要字段如下：

| 输入字段 | 典型形状 | 含义 |
| --- | --- | --- |
| `/observations/qpos` | `(T, 14)` | 当前时刻机器人状态 |
| `/action` | `(T, 14)` | 与当前观测配对的下一时刻关节目标 |
| `/observations/images/cam_high` | `(T,)` JPEG 字节串 | 头部相机 |
| `/observations/images/cam_left_wrist` | `(T,)` JPEG 字节串 | 左腕相机 |
| `/observations/images/cam_right_wrist` | `(T,)` JPEG 字节串 | 右腕相机 |
| `/observations/qvel` | `(T, 14)`，可选 | 关节速度 |
| `/observations/effort` | `(T, 14)`，可选 | 关节力矩或 effort |

当前 `process_data.py` 不会产生 `qvel` 和 `effort`，所以 RoboTwin 的常规 pi05 数据只有
state、action 和三路 RGB 图像。

`instructions.json` 格式为：

```json
{
  "instructions": [
    "Use the right arm to grab and lift the bottle.",
    "Lift the bottle upright with the right arm."
  ]
}
```

## 4. LeRobot feature 映射

`create_empty_dataset()` 先声明目标数据集 schema。核心映射如下：

| 中间 HDF5 | LeRobot feature | 类型与形状 |
| --- | --- | --- |
| `/observations/qpos[i]` | `observation.state` | `float32, (14,)` |
| `/action[i]` | `action` | `float32, (14,)` |
| `cam_high[i]` | `observation.images.cam_high` | `image, (3, 480, 640)` |
| `cam_left_wrist[i]` | `observation.images.cam_left_wrist` | `image, (3, 480, 640)` |
| `cam_right_wrist[i]` | `observation.images.cam_right_wrist` | `image, (3, 480, 640)` |
| 随机选中的 instruction | `task`，写盘后用 `task_index` 表示 | 字符串任务描述 |

14 维 state/action 的顺序是：

```text
[
  left_waist,
  left_shoulder,
  left_elbow,
  left_forearm_roll,
  left_wrist_angle,
  left_wrist_rotate,
  left_gripper,
  right_waist,
  right_shoulder,
  right_elbow,
  right_forearm_roll,
  right_wrist_angle,
  right_wrist_rotate,
  right_gripper
]
```

即使任务名是 `adjust_bottle_singlearm`，当前数据格式仍然保存左右两条机械臂的完整 14 维
状态和动作。`singlearm` 表示任务主要使用单臂完成，并不表示数据维度缩减为 7。

## 5. 各函数的转换逻辑

### 5.1 `create_empty_dataset()`

这个函数创建空的 `LeRobotDataset`：

- 声明 14 个 motor 名称及其顺序。
- 声明三路相机名称。
- 创建 `observation.state` 和 `action` feature。
- 如果输入包含 `qvel` 或 `effort`，增加对应的可选 feature。
- 将数据集 fps 固定为 50 Hz。
- 如果 `HF_LEROBOT_HOME/repo_id` 已存在，先删除旧目录。

fps 会影响 LeRobot 自动生成的时间戳：

```text
timestamp[i] = i / 50
```

如果以后采集频率不是 50 Hz，需要同步修改这里的 `fps=50`。

### 5.2 `load_raw_images_per_camera()`

函数兼容两种图片存储方式：

- 如果相机 dataset 是四维数组 `(T, H, W, C)`，直接读取。
- 如果是一维 JPEG 字节串数组 `(T,)`，使用 `cv2.imdecode()` 逐帧解码。

由当前 `process_data.py` 生成的数据属于第二种。图片在上一阶段已经统一为 640×480，当前
函数不再 resize，也不进行像素归一化。

该函数会一次性把一条 episode 的指定相机图片全部放入内存。长轨迹或高分辨率数据可能
产生较高的瞬时内存占用，但处理完当前 episode 后才会进入下一条。

### 5.3 `load_raw_episode_data()`

函数读取一条 episode：

```text
/observations/qpos  -> torch.Tensor state
/action             -> torch.Tensor action
三路相机             -> dict[str, np.ndarray]
qvel/effort          -> 存在时读取，否则为 None
```

相机列表当前是硬编码的：

```text
cam_high
cam_left_wrist
cam_right_wrist
```

虽然脚本中另有 `get_cameras()` 可以发现相机名称，但主流程没有调用它。因此缺少任意一路
相机都会报错；增加或更换相机时，也需要同步修改 feature 定义、读取列表和训练配置。

### 5.4 `populate_dataset()`

这是逐 episode、逐 frame 写数据的核心函数。

对每条 episode：

1. 调用 `load_raw_episode_data()` 读取全部 state、action 和图片。
2. 打开同目录的 `instructions.json`。
3. 用 `np.random.choice()` 随机选择一条 instruction。
4. 为每一帧构造 frame 字典。
5. 调用 `dataset.add_frame(frame)` 加入当前 episode buffer。
6. 全部帧加入后调用 `dataset.save_episode()` 写盘。

一帧在加入 LeRobot 前大致如下：

```python
frame = {
    "observation.state": state[i],
    "action": action[i],
    "task": instruction,
    "observation.images.cam_high": cam_high[i],
    "observation.images.cam_left_wrist": cam_left_wrist[i],
    "observation.images.cam_right_wrist": cam_right_wrist[i],
}
```

同一条 episode 的全部 frame 使用同一条 instruction。脚本没有设置随机种子，因此使用
相同输入重复转换时，每条 episode 随机得到的 instruction 可能不同。

`add_frame()` 还会自动增加：

- `frame_index`：当前 episode 内从 0 开始的帧编号。
- `timestamp`：按 `frame_index / fps` 计算的时间戳。

`save_episode()` 会增加或更新：

- `episode_index`
- 全局 `index`
- `task_index`
- 每条 episode 的统计信息

语言文本本身登记在 `meta/tasks.jsonl`，帧数据保存对应的 `task_index`。训练配置中的
`prompt_from_task=True` 会根据这个索引取回任务文本，作为 pi05 的 prompt。

### 5.5 `port_aloha()`

这是命令行入口，流程如下：

```text
检查并删除旧输出目录
        ↓
检查 raw_dir
        ↓
递归寻找 *.hdf5
        ↓
根据第一条 HDF5 判断是否包含 qvel/effort
        ↓
create_empty_dataset()
        ↓
populate_dataset()
        ↓
可选 push_to_hub()
```

默认只生成本地数据集，不会上传。只有显式加入 `--push-to-hub` 才会调用 Hugging Face Hub
上传逻辑。

## 6. 时间对齐在本阶段如何处理

时间对齐已经由上一阶段 `process_data.py` 完成：

```text
LeRobot frame k:
  observation.state = 原始 state[k]
  images             = 原始 images[k]
  action             = 原始 state[k + 1]
```

因此，原始 episode 如果有 N 帧：

- `process_data.py` 输出 N-1 个中间样本。
- 当前 LeRobot 转换仍输出 N-1 个 frame。

例如当前 `adjust_bottle_singlearm/demo_clean` 的原始 `episode0` 有 97 帧，第一阶段会产生
96 个 state/action/image 配对，第二阶段仍会保存 96 个 LeRobot frame。

pi05 训练所需的未来 action chunk 不是在这个转换脚本中预先展开的。训练数据加载器会依据
数据集 fps 和模型的 `action_horizon`，从连续的单步 `action` 中查询未来动作序列。

## 7. 默认输出结构

使用默认 `mode="image"` 时，主要输出类似：

```text
~/.cache/huggingface/lerobot/<repo_id>/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── meta/
    ├── info.json
    ├── tasks.jsonl
    ├── episodes.jsonl
    └── episodes_stats.jsonl
```

- `info.json`：fps、robot_type、feature schema、总帧数和总 episode 数等。
- `tasks.jsonl`：自然语言任务文本与 `task_index` 的映射。
- `episodes.jsonl`：每条 episode 的长度和任务。
- `episodes_stats.jsonl`：逐 episode 的特征统计。
- `episode_*.parquet`：逐帧 state、action、图片及索引信息。

LeRobot 数据格式随项目依赖版本变化，应以生成后的 `meta/info.json` 中
`codebase_version` 为准，不要只根据转换脚本文件名或旧版教程判断格式版本。

## 8. 主要命令行参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--raw-dir` | 必填 | 中间 episode 根目录 |
| `--repo-id` | 必填 | 本地数据集 ID，也是默认输出相对路径 |
| `--episodes` | 全部 | 只转换指定的 HDF5 列表下标 |
| `--push-to-hub` | `False` | 转换完成后上传到 Hugging Face Hub |
| `--is-mobile` | `False` | 将 robot_type 从 `aloha` 改为 `mobile_aloha` |
| `--mode` | `image` | 相机 feature 使用 `image` 或 `video` |
| `--task` | `DEBUG` | 当前函数接收但未实际用于写入数据 |
| `--raw-repo-id` | `None` | 原计划用于下载原始数据，当前下载代码已注释 |

只转换选定条目示例：

```bash
uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
  --raw-dir processed_data/adjust_bottle_singlearm-demo_clean-100 \
  --repo-id adjust_bottle_singlearm_debug \
  --episodes 0 1 2
```

注意：`--episodes 0 1 2` 指的是脚本扫描得到的 `hdf5_files` 列表下标，不是可靠地按
`episode_0.hdf5`、`episode_1.hdf5`、`episode_2.hdf5` 文件名匹配。

## 9. 当前实现中的重要注意事项

### 9.1 会删除同名输出目录

如果 `~/.cache/huggingface/lerobot/<repo_id>` 已存在，脚本会使用 `shutil.rmtree()` 删除整个
目录后重新生成。运行前应确认 `repo_id`，不要把仍需保留的数据集作为目标。

### 9.2 HDF5 文件没有排序

脚本通过 `os.walk()` 和 `fnmatch` 收集文件，但没有执行排序。因此 LeRobot 的
`episode_index=0` 不一定来自 `episode_0.hdf5`。这通常不影响训练，因为 episode 本身是
独立样本；但会影响问题排查、指定 `--episodes` 和原始轨迹追踪。

### 9.3 schema 只根据第一条 HDF5 判断

`qvel` 和 `effort` 是否存在只检查 `hdf5_files[0]`。所有 episode 应保持完全一致的字段
结构，否则可能在处理后续 episode 时失败。

### 9.4 `raw_repo_id` 当前不能完成下载

下载函数的 import 和调用都已注释。如果 `raw_dir` 不存在但传入了 `raw_repo_id`，程序也
不会真正下载文件，之后仍会因找不到 HDF5 失败。实际使用时应确保 `raw_dir` 已存在。

### 9.5 `task` 参数当前无效

虽然 CLI 提供 `--task`，`populate_dataset()` 最终写入的是 `instructions.json` 中随机选出的
文本。修改 `--task` 不会覆盖这些指令。

### 9.6 固定假设

当前脚本固定假设：

- 14 维双臂状态和动作。
- 三路相机都存在。
- 图像分辨率为 640×480。
- 数据频率为 50 Hz。
- 每个 HDF5 旁边都有非空的 `instructions.json`。

更换机器人、相机或采样频率时，这些位置需要一起检查。

## 10. 转换后如何接入训练

完成转换后，至少需要修改 `policy/pi05/src/openpi/training/config.py` 中目标训练配置的
`repo_id`：

```python
data=LeRobotAlohaDataConfig(
    repo_id="adjust_bottle_singlearm",
    # ...
)
```

并确保 repack 映射与本数据集 feature 一致：

```python
{
    "images": {
        "cam_high": "observation.images.cam_high",
        "cam_left_wrist": "observation.images.cam_left_wrist",
        "cam_right_wrist": "observation.images.cam_right_wrist",
    },
    "state": "observation.state",
    "actions": "action",
    "prompt": "prompt",
}
```

之后先计算 normalization statistics：

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi05_base_adjust_bottle_singlearm_lora
```

再启动训练：

```bash
bash finetune.sh \
  pi05_base_adjust_bottle_singlearm_lora \
  adjust_bottle_singlearm_lora \
  0,1
```

其中训练配置名、实验名和 GPU 编号应按实际环境调整。
