import sys

sys.path.append("./policy/ACT/")

import os
import h5py
import numpy as np
import pickle
import cv2
import argparse
import pdb
import json


def load_hdf5(dataset_path):
    # 读取 RoboTwin 原始 episode 文件。
    # HDF5 可以粗略理解成“一个文件里的小型文件系统”：
    # - group 类似文件夹，例如 /joint_action、/observation/head_camera
    # - dataset 类似保存 ndarray 的文件，例如 /joint_action/left_arm
    # - dataset 后面的 [()] 表示一次性把整个数组读进内存。
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        # 读取左右臂和夹爪的专家动作/状态序列。
        # 这些路径对应原始 HDF5 内部的层级，不是普通磁盘路径。
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        image_dict = dict()
        # /observation 下每个 camera 是一个 group，里面的 rgb 是该相机的图像序列。
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def images_encoding(imgs):
    # 把图像列表编码成 JPEG 字节串，并补齐到同样长度。
    # 当前 data_transform 里实际没有使用这个函数，而是直接保存 uint8 图像数组。
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def data_transform(path, episode_num, save_path):
    # 将 RoboTwin 原始数据目录中的 episode{i}.hdf5 转成 ACT 所需格式：
    #
    # 输出文件结构大致是：
    # episode_i.hdf5
    # ├── action                         # shape: [T, action_dim]，下一时刻要执行的动作
    # └── observations
    #     ├── qpos                       # shape: [T, action_dim]，当前时刻机器人关节状态
    #     ├── left_arm_dim               # 每帧左臂关节维度，通常固定
    #     ├── right_arm_dim              # 每帧右臂关节维度，通常固定
    #     └── images
    #         ├── cam_high               # 头部相机图像
    #         ├── cam_right_wrist        # 右腕相机图像
    #         └── cam_left_wrist         # 左腕相机图像
    begin = 0
    floders = os.listdir(path)
    assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(episode_num):
        # 每个 episode 是一条专家演示轨迹；这里先从原始 HDF5 中读出关节和图像。
        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = (load_hdf5(
            os.path.join(path, f"episode{i}.hdf5")))
        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []

        last_state = None
        for j in range(0, left_gripper_all.shape[0]):
            # 第 j 帧的左右臂关节和夹爪值。
            # left_arm/right_arm 是多个关节角，left_gripper/right_gripper 是单个夹爪开合值。

            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )

            if j != left_gripper_all.shape[0] - 1:
                # ACT 中的 qpos 表示“当前观测时刻的机器人状态”。
                # 这里拼成一个向量：[左臂关节, 左夹爪, 右臂关节, 右夹爪]。
                state = np.concatenate((left_arm, [left_gripper], right_arm, [right_gripper]), axis=0)  # joint

                state = state.astype(np.float32)
                qpos.append(state)

                # 原始 HDF5 里的 rgb 是压缩后的图片字节；先解码成 OpenCV 图像，再统一缩放到 640x480。
                camera_high_bits = image_dict["head_camera"][j]
                camera_high = cv2.imdecode(np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_high_resized = cv2.resize(camera_high, (640, 480))
                cam_high.append(camera_high_resized)

                camera_right_wrist_bits = image_dict["right_camera"][j]
                camera_right_wrist = cv2.imdecode(np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_right_wrist_resized = cv2.resize(camera_right_wrist, (640, 480))
                cam_right_wrist.append(camera_right_wrist_resized)

                camera_left_wrist_bits = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_left_wrist_resized = cv2.resize(camera_left_wrist, (640, 480))
                cam_left_wrist.append(camera_left_wrist_resized)

            if j != 0:
                # action 表示“从上一帧观测出发，下一步应该到达的状态”。
                # 因此 qpos 使用第 0 到倒数第 2 帧，action 使用第 1 到最后一帧，
                # 两者长度对齐后就是监督学习样本：qpos[j] -> action[j]。
                action = state
                actions.append(action)
                left_arm_dim.append(left_arm.shape[0])
                right_arm_dim.append(right_arm.shape[0])

        hdf5path = os.path.join(save_path, f"episode_{i}.hdf5")

        with h5py.File(hdf5path, "w") as f:
            # create_dataset 会在 HDF5 文件里创建一个数组节点。
            # 这里 action 放在根目录下，对应读取路径 "/action"。
            f.create_dataset("action", data=np.array(actions))
            # create_group 会创建“文件夹”节点，后续 dataset 都挂在这个 group 下面。
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            image = obs.create_group("images")
            # cam_high_enc, len_high = images_encoding(cam_high)
            # cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
            # cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)
            # 三路相机图像保存为未压缩 uint8 数组，形状通常是 [T, 480, 640, 3]。
            image.create_dataset("cam_high", data=np.stack(cam_high), dtype=np.uint8)
            image.create_dataset("cam_right_wrist", data=np.stack(cam_right_wrist), dtype=np.uint8)
            image.create_dataset("cam_left_wrist", data=np.stack(cam_left_wrist), dtype=np.uint8)

        begin += 1
        print(f"proccess {i} success!")

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., adjust_bottle)",
    )
    parser.add_argument("task_config", type=str)
    parser.add_argument("expert_data_num", type=int)

    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    expert_data_num = args.expert_data_num

    # 原始数据路径来自 collect_data.sh 采集结果：
    # ../../data/{task_name}/{task_config}/data/episode{i}.hdf5
    # 转换后的数据保存到 policy/ACT/processed_data 下，供 ACT dataloader 读取。
    begin = 0
    begin = data_transform(
        os.path.join("../../data/", task_name, task_config, 'data'),
        expert_data_num,
        f"processed_data/sim-{task_name}/{task_config}-{expert_data_num}",
    )

    SIM_TASK_CONFIGS_PATH = "./SIM_TASK_CONFIGS.json"

    # 更新 ACT 的任务配置文件，让训练脚本知道新数据集的位置、episode 数量和相机名。
    try:
        with open(SIM_TASK_CONFIGS_PATH, "r") as f:
            SIM_TASK_CONFIGS = json.load(f)
    except Exception:
        SIM_TASK_CONFIGS = {}

    SIM_TASK_CONFIGS[f"sim-{task_name}-{task_config}-{expert_data_num}"] = {
        "dataset_dir": f"./processed_data/sim-{task_name}/{task_config}-{expert_data_num}",
        "num_episodes": expert_data_num,
        "episode_len": 1000,
        "camera_names": ["cam_high", "cam_right_wrist", "cam_left_wrist"],
    }

    with open(SIM_TASK_CONFIGS_PATH, "w") as f:
        json.dump(SIM_TASK_CONFIGS, f, indent=4)
