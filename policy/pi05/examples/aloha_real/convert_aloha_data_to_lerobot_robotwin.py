"""
Convert the ALOHA-style HDF5 files produced by RoboTwin into a LeRobot dataset.

Run from policy/pi05, for example:
uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
    --raw-dir processed_data/adjust_bottle_singlearm-demo_clean-100 \
    --repo-id adjust_bottle_singlearm_repo

See docs/pi05_RoboTwin中间数据转LeRobot说明.md for the complete data mapping.
"""

# 本脚本在 RoboTwin -> pi05 数据链路中负责第二阶段转换：
#
#   RoboTwin 原始 HDF5
#       -- scripts/process_data.py --> ALOHA 风格的逐 episode HDF5
#       -- 本脚本                  --> LeRobot 数据集
#
# 推荐从 policy/pi05 目录运行。例如：
#   bash generate.sh \
#       processed_data/adjust_bottle_singlearm-demo_clean-100 \
#       adjust_bottle_singlearm_repo
#
# 默认输出位置不是 raw_dir 旁边，而是：
#   HF_LEROBOT_HOME / repo_id
# 在当前默认环境中，HF_LEROBOT_HOME 通常为 ~/.cache/huggingface/lerobot。

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro
import json
import os
import fnmatch


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    # LeRobot 数据集写盘参数。frozen=True 表示实例创建后字段不可直接修改；Tyro 会把
    # 这些字段展开为 --dataset-config.* 命令行参数。
    #
    # use_videos 控制 LeRobot 是否启用视频能力；实际相机 feature 的 dtype 由下方
    # create_empty_dataset() 的 mode 参数决定。当前 port_aloha() 默认 mode="image"。
    use_videos: bool = True

    # LeRobot 会根据 fps 自动生成 timestamp，此值是检查时间戳同步时允许的误差（秒）。
    tolerance_s: float = 0.0001

    # add_frame() 会先把图像写入临时位置；这里配置异步图片写入进程数和线程数。
    image_writer_processes: int = 10
    image_writer_threads: int = 5

    # 仅在视频模式下需要指定特殊视频后端；None 表示使用 LeRobot 默认后端。
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    # 创建一个还没有 frame/episode 的 LeRobot 数据集，并声明每帧的数据 schema。
    # repo_id 既是数据集标识，也参与组成本地输出路径 HF_LEROBOT_HOME/repo_id。

    # 这 14 个名称的顺序必须与 scripts/process_data.py 中 state/action 的拼接顺序完全一致：
    #   左臂 6 关节 + 左夹爪 + 右臂 6 关节 + 右夹爪。
    # 名称只描述每个数值的含义，不会在这里重新排列实际数组。
    motors = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    ]

    # pi05 的当前 RoboTwin/ALOHA 配置固定消费三路相机。这里的名字同时必须与输入 HDF5
    # 的 /observations/images/<camera> 以及训练配置中的 repack_transforms 对齐。
    cameras = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    # features 是 LeRobot 对“每一帧包含哪些字段”的声明。
    # state 和 action 都是 14 维 float32；process_data.py 已经把它们错开一帧：
    # observation.state[k] 是原轨迹 state[k]，action[k] 是原轨迹 state[k + 1]。
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        },
    }

    # qvel 和 effort 是可选字段。RoboTwin 当前 process_data.py 不生成它们，因此通常不会
    # 进入这两个分支；保留该逻辑是为了兼容包含速度或力矩信息的 ALOHA 数据。
    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        }

    # 为三路相机逐一声明 feature。shape 使用 LeRobot 的 (C, H, W) schema 写法，
    # 传给 add_frame() 的实际 NumPy 图片则是 OpenCV 解码出的 (H, W, C)。
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    # 转换不是增量追加：如果同名 repo_id 已存在，会先递归删除整个旧数据集。
    # 因此重复运行同一命令会从头生成，而不是接着已有 episode 继续写。
    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    # fps 固定为 50 Hz。LeRobot 据此为第 i 帧生成 timestamp=i/50；若采集频率改变，
    # 这里也应同步修改，否则时间戳不会反映真实采样间隔。
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    # 从第一条 episode 自动发现非 depth 相机的辅助函数。
    # 当前主流程没有调用它：load_raw_episode_data() 仍然固定读取三路指定相机。
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(hdf5_files: list[Path]) -> bool:
    # 只检查第一条 episode 来确定整个数据集是否声明 velocity feature。
    # 这隐含要求所有 episode 的字段结构一致。
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    # 与 has_velocity 相同，只用第一条 episode 判断是否存在 effort。
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    # 把一条 episode 的指定相机图像全部载入内存。
    # 返回格式：{camera_name: np.ndarray[T, H, W, C]}。
    imgs_per_cam = {}
    for camera in cameras:
        # 未压缩图片通常直接以 (T, H, W, C) 四维数组存储；process_data.py 生成的是
        # (T,) 定长 JPEG 字节串，所以 RoboTwin 数据通常走下面的解码分支。
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            # cv2 只在需要解码 JPEG 时局部导入，避免处理纯数组图片时依赖该模块。
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                # HDF5 中的一帧是 bytes/np.bytes_。先将其解释为 uint8 字节流，再交给
                # OpenCV 解码；定长字符串尾部可能存在的 \0 不影响 JPEG 解码。
                data = np.frombuffer(data, np.uint8)
                # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 解码为彩色图像
                imgs_array.append(cv2.imdecode(data, cv2.IMREAD_COLOR))
            imgs_array = np.array(imgs_array)

        # OpenCV 解码得到的单帧通常是 uint8、(480, 640, 3)。代码没有在这里做 resize、
        # 通道变换或归一化；尺寸统一工作已经由上一阶段 process_data.py 完成。
        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[
        dict[str, np.ndarray],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
    # 一次性读取单条中间 HDF5 的全部训练字段。
    # state/action 转成 torch.Tensor 是因为 LeRobot frame schema 接受 Tensor；图像仍保留
    # NumPy 数组。文件关闭后这些数据都已经位于内存中，可以继续安全使用。
    with h5py.File(ep_path, "r") as ep:
        # process_data.py 输出的 qpos/action 已经是长度相同的 (T, 14) float32 数组。
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])

        # 速度和力矩按“存在才读取”的方式兼容可选输入。
        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        # 虽然上面定义了 get_cameras()，当前实际读取列表仍固定为这三路；缺少任意一路
        # 都会在访问 HDF5 key 时抛出异常。
        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [
                "cam_high",
                "cam_left_wrist",
                "cam_right_wrist",
            ],
        )

    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    # 将若干 HDF5 episode 逐帧填入已经创建好的 LeRobotDataset。
    # 注意：task 参数当前没有在函数体中使用；真正写入 frame["task"] 的文本来自每个
    # episode 目录中的 instructions.json。
    if episodes is None:
        # 默认转换扫描到的全部文件。若传入 --episodes，则其中的数字是 hdf5_files 列表
        # 下标，不一定等于原文件名中的 episode 编号。
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        # 每处理一条 episode，就把该条轨迹的状态、动作和三路图像完整加载到内存。
        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]

        # instructions.json 与 episode_*.hdf5 位于同一目录，由 scripts/process_data.py 生成。
        # 每条 episode 从候选 instructions 中随机选一条，而且只选一次，因此该 episode
        # 的所有 frame 共用同一个自然语言任务描述。代码没有设置随机种子，多次转换时
        # 同一 episode 可能得到不同指令。
        dir_path = os.path.dirname(ep_path)
        json_Path = f"{dir_path}/instructions.json"

        with open(json_Path, 'r') as f_instr:
            instruction_dict = json.load(f_instr)
            instructions = instruction_dict['instructions']
            instruction = np.random.choice(instructions)

        for i in range(num_frames):
            # 组装 LeRobot 的一帧。task 不需要预先出现在 features 中：LeRobot 会在
            # save_episode() 时将文本登记到 meta/tasks.jsonl，并在帧表中保存 task_index。
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": instruction,
            }

            # 将相机字典扁平化成 LeRobot feature key，例如：
            # observation.images.cam_high、observation.images.cam_left_wrist。
            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            # 可选 feature 只有在 create_empty_dataset() 已声明且本条 episode 已读取时才写入。
            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]

            # add_frame() 会校验字段及形状，并自动补充 frame_index 和 timestamp。
            # 它先把数据加入当前 episode buffer；完整 episode 的表格要到 save_episode()
            # 才正式写出。
            dataset.add_frame(frame)

        # 将当前 buffer 保存为一个 episode（默认 image 模式下主要写成 Parquet 和元数据），
        # 更新 episode 统计信息，然后清空/重建 buffer 供下一条轨迹使用。
        dataset.save_episode()

    return dataset


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    # 转换总入口，由最下方 tyro.cli() 暴露为命令行接口。

    # 与 create_empty_dataset() 内部的检查重复，但作用相同：输出目录已存在时整目录删除。
    # 该删除发生在检查 raw_dir 之前，因此 repo_id 一定要确认无误，避免覆盖已有数据集。
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    # raw_repo_id 原本用于在本地原始目录不存在时下载数据，但下载调用目前已被注释。
    # 因而实际使用时 raw_dir 必须存在；仅提供 raw_repo_id 并不能完成下载。
    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        # download_raw(raw_dir, repo_id=raw_repo_id)

    # 递归收集 raw_dir 下全部 *.hdf5。这里没有排序，所以 hdf5_files 的顺序取决于
    # os.walk/filesystem 返回顺序；最终 LeRobot episode_index 按此列表顺序重新从 0 编号。
    # raw_dir 中若没有 HDF5，后续 has_effort()/has_velocity() 访问第一个元素时会报错。
    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            file_path = os.path.join(root, filename)
            hdf5_files.append(file_path)

    # 可选字段只依据 hdf5_files[0] 判断。robot_type 默认写为 aloha；--is-mobile 会改成
    # mobile_aloha。mode 决定图像 feature 采用 image 还是 video 格式。
    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
    )

    # 把指定 episode 的所有 frame 加入 LeRobotDataset，并逐 episode 写盘。
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
    )
    # dataset.consolidate()

    # 默认只保存在本地；只有显式传入 --push-to-hub 才会上传到 Hugging Face Hub。
    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    # Tyro 根据 port_aloha() 的参数名和类型注解自动生成 CLI，例如：
    #   --raw-dir PATH --repo-id NAME --episodes 0 1 2 --push-to-hub
    tyro.cli(port_aloha)
