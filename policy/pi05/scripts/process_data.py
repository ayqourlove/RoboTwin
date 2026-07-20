# RoboTwin 原始数据 -> pi05/ALOHA 中间格式转换脚本。
#
# 数据处理链路分为两步：
#   1. 本脚本读取 data/<task_name>/<setting>/ 下的 RoboTwin HDF5 和语言指令，
#      将每条轨迹转换为 pi05 后续转换程序能够读取的 ALOHA 风格 HDF5；
#   2. examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py 再把这里的
#      中间结果转换为训练实际使用的 LeRobot 数据集。
#
# 以仓库中的 adjust_bottle_singlearm/demo_clean 数据为例，应在 policy/pi05 目录运行：
#   bash process_data_pi05.sh adjust_bottle_singlearm demo_clean 100
#
# 输入目录结构：
#   ../../data/adjust_bottle_singlearm/demo_clean/
#   ├── data/episode0.hdf5
#   ├── data/episode1.hdf5
#   └── instructions/episode0.json, episode1.json, ...
#
# 输出目录结构：
#   processed_data/adjust_bottle_singlearm-demo_clean-100/
#   ├── episode_0/
#   │   ├── episode_0.hdf5
#   │   └── instructions.json
#   └── episode_1/...
#
# 注意：脚本中的输入、输出都是相对路径，因此运行时的当前目录应为 policy/pi05。

import sys

import os
import h5py
import numpy as np
import pickle
import cv2
import argparse
import yaml, json


def load_hdf5(dataset_path):
    # 读取一条 RoboTwin 原始轨迹。
    #
    # 原始文件中本脚本会使用以下字段：
    #   /joint_action/left_arm           (N, 6)
    #   /joint_action/left_gripper       (N,)
    #   /joint_action/right_arm          (N, 6)
    #   /joint_action/right_gripper      (N,)
    #   /observation/<camera_name>/rgb   (N,)，每个元素是一帧 JPEG 字节串
    #
    # N 是本条 episode 的帧数。数组使用 [()] 一次性读入内存，因此 with 块结束、
    # HDF5 文件关闭后，返回的数据仍然可用。
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        # 这里读取的是采集时记录的机器人关节状态。RoboTwin 的当前双臂布局为：
        # 左臂 6 个关节 + 左夹爪 1 维 + 右臂 6 个关节 + 右夹爪 1 维。
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )

        # 保存所有相机的 RGB 压缩数据，键名沿用原始 HDF5 中的相机名。
        # 当前转换逻辑随后明确要求存在 head_camera、left_camera、right_camera。
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def images_encoding(imgs):
    # 将一组已经解码并 resize 的 OpenCV 图像重新压缩成 JPEG 字节串。
    # 输入 imgs 是长度为 T 的图像列表，每帧形状通常为 (480, 640, 3)。
    #
    # HDF5 的定长字符串 dtype（S<max_len>）要求同一个 dataset 中的元素宽度一致，
    # 因此需要先遍历全部图片，找出该相机所有 JPEG 中的最大字节数 max_len。
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        # cv2.imencode 返回 (是否成功, 编码后的 uint8 数组)。当前代码保留 success，
        # 但没有单独处理编码失败；正常情况下 encoded_image.tobytes() 得到 JPEG 数据。
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))

    # 构造过手动补零的数据，但当前函数最终返回的是 encode_data，而不是 padded_data。
    # 调用方通过 dtype=f"S{max_len}" 创建 HDF5 dataset 时，h5py 会按定长字符串存储；
    # 因此这里的 padded_data 目前只是保留下来的中间变量，不参与最终写盘。
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def get_task_config(task_name):
    # 读取任务 YAML 配置的辅助函数。当前脚本的主流程没有调用它，保留用于兼容
    # 其他 policy 的同类处理脚本或后续扩展。
    with open(f"./task_config/{task_name}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return args


def data_transform(path, episode_num, save_path):
    # 批量转换前 episode_num 条轨迹。
    #
    # 参数：
    #   path        原始设置目录，例如 ../../data/adjust_bottle_singlearm/demo_clean
    #   episode_num 要处理的 episode 数量，程序假定编号从 0 连续到 episode_num - 1
    #   save_path   中间数据输出目录，例如
    #               processed_data/adjust_bottle_singlearm-demo_clean-100
    #
    # 返回值 begin 是成功走完循环的 episode 数量。函数没有跳过坏数据的逻辑：只要某个
    # HDF5/JSON 缺失或字段不符合预期，就会直接抛出异常并停止，之前写完的数据会保留。
    begin = 0

    # os.listdir 会在输入目录不存在时尽早报错。floders（原代码拼写）及下方断言目前未使用，
    # 所以 episode_num 是否超过实际数据量，最终由逐个打开 episode 文件时的错误体现。
    floders = os.listdir(path)
    # assert episode_num <= len(floders), "data num not enough"

    # exist_ok 风格的等价写法：目录不存在才创建；已有目录不会被整体清空。
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(episode_num):

        # RoboTwin 的指令 JSON 通常按 seen/unseen 分类。训练数据固定选取 seen 列表，
        # 并改写成下游 LeRobot 转换脚本所期待的 {"instructions": [...]} 形式。
        desc_type = "seen"
        instruction_data_path = os.path.join(path, "instructions", f"episode{i}.json")
        with open(instruction_data_path, "r") as f_instr:
            instruction_dict = json.load(f_instr)
        instructions = instruction_dict[desc_type]
        save_instructions_json = {"instructions": instructions}

        # 每条轨迹单独占用一个目录。exist_ok=True 允许重新运行；同名文件随后会被覆盖。
        os.makedirs(os.path.join(save_path, f"episode_{i}"), exist_ok=True)

        with open(
                os.path.join(os.path.join(save_path, f"episode_{i}"), "instructions.json"),
                "w",
        ) as f:
            json.dump(save_instructions_json, f, indent=2)

        # 原始轨迹文件名没有下划线（episode0.hdf5），输出文件名有下划线
        #（episode_0.hdf5），这是两种数据格式各自的命名约定。
        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = (load_hdf5(
            os.path.join(path, "data", f"episode{i}.hdf5")))

        # qpos 和三路相机图像组成模型在时刻 t 看到的 observation；actions 是监督标签。
        # 下方通过错开一帧构造 (observation[t], action[t+1])，所以长度为 N 的原始轨迹
        # 会产生 N-1 个训练样本。
        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []

        # 当前变量未参与后续逻辑，属于历史实现遗留。
        last_state = None
        for j in range(0, left_gripper_all.shape[0]):

            # 取出原始轨迹第 j 帧的左右臂关节角和夹爪状态。
            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )

            # 固定拼接顺序如下：
            #   [左臂 6 关节, 左夹爪, 右臂 6 关节, 右夹爪]
            # 对当前 ALOHA/AgileX 机器人而言一共是 14 维。singlearm 任务仍然保留完整
            # 双臂状态，只是示范动作主要由其中一条手臂完成。
            state = np.array(left_arm.tolist() + [left_gripper] + right_arm.tolist() + [right_gripper])  # joints angle

            # pi05/LeRobot 侧的 observation.state 和 action feature 均声明为 float32。
            state = state.astype(np.float32)

            # 最后一帧没有“下一帧动作”可配对，因此 qpos 和图像只保存第 0..N-2 帧。
            if j != left_gripper_all.shape[0] - 1:
                qpos.append(state)

                # 原始 HDF5 中每帧 RGB 是 JPEG 字节串：先还原为 OpenCV 图像，统一缩放到
                # 640x480，再在写输出 HDF5 时重新编码为 JPEG。
                # cv2.resize 的 size 参数顺序是 (宽, 高)，不是 (高, 宽)。
                camera_high_bits = image_dict["head_camera"][j]
                camera_high = cv2.imdecode(np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_high_resized = cv2.resize(camera_high, (640, 480))
                cam_high.append(camera_high_resized)

                # 原始 right_camera 在中间格式中重命名为 cam_right_wrist。
                camera_right_wrist_bits = image_dict["right_camera"][j]
                camera_right_wrist = cv2.imdecode(np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_right_wrist_resized = cv2.resize(camera_right_wrist, (640, 480))
                cam_right_wrist.append(camera_right_wrist_resized)

                # 原始 left_camera 在中间格式中重命名为 cam_left_wrist。
                camera_left_wrist_bits = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_left_wrist_resized = cv2.resize(camera_left_wrist, (640, 480))
                cam_left_wrist.append(camera_left_wrist_resized)

            # 第 0 帧没有对应的前一时刻 observation，因此 action 保存第 1..N-1 帧状态。
            # 写盘后，同一下标 k 表示：
            #   observations/qpos[k] = state[k]
            #   observations/images/*[k] = image[k]
            #   action[k] = state[k + 1]
            # 即用当前观测预测下一时刻关节目标。
            if j != 0:
                action = state
                actions.append(action)

                # 记录每个时间步左右机械臂的关节维数。当前数据通常始终为 6；这两个字段
                # 属于中间格式的辅助元数据，后续 RoboTwin LeRobot 转换脚本并不读取它们。
                left_arm_dim.append(left_arm.shape[0])
                right_arm_dim.append(right_arm.shape[0])

        hdf5path = os.path.join(save_path, f"episode_{i}/episode_{i}.hdf5")

        # 以 "w" 模式创建输出：若该 episode 的 HDF5 已存在，会被覆盖。
        # 主要输出结构为：
        #   /action                         (N-1, 14)
        #   /observations/qpos              (N-1, 14)
        #   /observations/left_arm_dim      (N-1,)
        #   /observations/right_arm_dim     (N-1,)
        #   /observations/images/cam_high   (N-1,) JPEG 字节串
        #   /observations/images/cam_left_wrist
        #   /observations/images/cam_right_wrist
        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(actions))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            image = obs.create_group("images")

            # 每个相机分别统计最大 JPEG 长度，因为三路相机的图像内容和压缩后大小不同。
            cam_high_enc, len_high = images_encoding(cam_high)
            cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
            cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)

            # S<长度> 是 HDF5/NumPy 的定长字节字符串类型。下游转换程序会逐帧读取这些
            # 字节，并用 cv2.imdecode 恢复为 (480, 640, 3) 图像。
            image.create_dataset("cam_high", data=cam_high_enc, dtype=f"S{len_high}")
            image.create_dataset("cam_right_wrist", data=cam_right_wrist_enc, dtype=f"S{len_right}")
            image.create_dataset("cam_left_wrist", data=cam_left_wrist_enc, dtype=f"S{len_left}")

        begin += 1
        # 保留原有输出文字（其中 proccess 是原代码拼写），每完成一条轨迹打印一次。
        print(f"proccess {i} success!")

    return begin


if __name__ == "__main__":
    # 三个参数都是位置参数。虽然 task_name 和 expert_data_num 声明了 default，
    # 但没有设置 nargs="?"，所以直接调用本脚本时仍需显式传入全部三个参数。
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        default="beat_block_hammer",
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("setting", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        default=50,
        help="Number of episodes to process (e.g., 50)",
    )
    args = parser.parse_args()

    task_name = args.task_name
    setting = args.setting
    expert_data_num = args.expert_data_num

    # 该相对路径以运行命令时的当前工作目录为基准，而不是以本 .py 文件的位置为基准。
    # 因此推荐从 policy/pi05 运行，此时 ../../data 正好指向 RoboTwin 仓库根目录下的 data。
    load_dir = os.path.join("../../data", str(task_name), str(setting))

    begin = 0
    # 这行只负责展示即将读取的路径；外层再拼接 "data" 不会影响实际的 load_dir。
    print(f'read data from path:{os.path.join("data", load_dir)}')

    # 转换结果先保存在 policy/pi05/processed_data 下，供后续 LeRobot 转换脚本读取。
    target_dir = f"processed_data/{task_name}-{setting}-{expert_data_num}"
    begin = data_transform(
        load_dir,
        expert_data_num,
        target_dir,
    )
