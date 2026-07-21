# RealMan pi05 远程推理接口

本文档用于机器人电脑（`10.147.17.38`）实现执行端。推理服务运行在 GPU 电脑
`10.147.17.223`，机器人端只负责采集状态、发送请求和安全执行动作。

## 1. 网络与启动

推理服务地址：

```text
WebSocket: ws://10.147.17.223:8000
Health:    http://10.147.17.223:8000/healthz
```

推理电脑启动命令：

```bash
cd /home/gzy/vla/RoboTwin/policy/pi05
bash serve_realman.sh
```

服务默认加载：

```text
train config: pi05_base_adjust_bottle_singlearm_lora
checkpoint:   checkpoints/pi05_base_adjust_bottle_singlearm_lora/
              adjust_bottle_singlearm_lora/10000
prompt:       Pick up the bottle and place it upright at the target position.
```

端口等参数可以覆盖，例如：

```bash
bash serve_realman.sh --port 8001 --action-horizon 5 --max-joint-delta 0.05
```

该服务没有身份认证，只应在受信任的机器人局域网中开放。

## 2. 客户端依赖

机器人端不需要安装 JAX、OpenPI 模型或 checkpoint，只需复制
`policy/pi05/packages/openpi-client` 并安装：

```bash
python3 -m pip install -e /path/to/openpi-client
```

依赖为 Python 3.7+、NumPy、msgpack 和 websockets。

## 3. 连接过程

使用 OpenPI 客户端：

```python
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(
    host="10.147.17.223",
    port=8000,
)
metadata = client.get_server_metadata()
```

连接成功后服务端首先发送 metadata。必须检查：

```python
assert metadata["protocol"] == "robotwin.realman.pi05.v1"
assert metadata["image_shape"] == [480, 640, 3]
assert metadata["action_horizon"] == 5
```

## 4. 请求格式

每次请求是一个由 msgpack-numpy 编码的 Python 字典：

```python
request = {
    "request_id": 123,
    "right_state": np.asarray(
        [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, gripper],
        dtype=np.float32,
    ),
    "images": {
        "middle": middle_rgb,
        "left": left_rgb,
        "right": right_rgb,
    },
}
response = client.infer(request)
```

字段约束：

| 字段 | 类型和形状 | 含义 |
| --- | --- | --- |
| `request_id` | `str` 或 `int`，可选 | 响应会原样返回，用于拒绝过期响应 |
| `right_state` | `float32[7]` | 右臂 6 个关节弧度，加夹爪归一化位置 |
| `images.middle` | `uint8[480,640,3]` | 头部 D435 彩色图 |
| `images.left` | `uint8[480,640,3]` | 左腕 D435 彩色图 |
| `images.right` | `uint8[480,640,3]` | 右腕 D435 彩色图 |

三路图像必须为 **RGB**，不能直接发送 OpenCV/ROS `bgr8` 数据。若通过
`cv_bridge` 得到 BGR，需要先执行：

```python
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
```

图像不得旋转、镜像或裁切，三路图像应尽可能使用相近时间戳。服务端会自行转换为
模型需要的 CHW 格式。

关节顺序必须和训练 URDF 中 `right_joint_1` 到 `right_joint_6` 一致，单位为弧度。
夹爪定义固定为：

```text
0.0 = 完全闭合
1.0 = 完全打开
```

服务端会从 checkpoint 的 `norm_stats.json` 读取固定左臂状态，并在右臂状态前补齐：

```text
[left_joint_1..6, left_gripper, right_joint_1..6, right_gripper]
```

机器人端不得自行再补一次左臂状态。

## 5. 响应格式

```python
{
    "request_id": 123,
    "actions": np.ndarray(shape=(5, 7), dtype=np.float32),
    "raw_action_horizon": 50,
    "policy_timing": {"infer_ms": ...},
    "server_timing": {"infer_ms": ..., "prev_total_ms": ...},
}
```

`actions` 每一行顺序为：

```text
[right_joint_1, right_joint_2, right_joint_3,
 right_joint_4, right_joint_5, right_joint_6, right_gripper]
```

前 6 维是**绝对关节目标角（rad）**，不是增量。最后一维是绝对夹爪目标 `[0,1]`。
服务端只返回模型预测的前 5 个动作，并基于请求中的当前关节状态，将相邻关节目标
变化限制在 `0.05 rad`。夹爪输出会裁剪到 `[0,1]`。

## 6. 推荐执行循环

机器人端建议按以下流程实现：

```text
1. 同步获取三路最新 RGB 和右臂当前状态。
2. 发送请求并等待响应。
3. 校验 request_id、数组形状、有限值和真实机器人关节限位。
4. 以 10 Hz 顺序执行 5 行绝对目标。
5. 重新采集观测并发起下一次推理。
```

伪代码：

```python
while not stopped:
    observation = collect_synchronized_observation()
    response = client.infer(observation)
    validate_response(response)

    for target in response["actions"]:
        publish_right_joint_target(target[:6])
        command_gripper(target[6])
        sleep_until_next_10hz_tick()
```

RealMan ROS1 驱动的 `embodied_arm_msgs/JointPos.joint` 接受弧度，并在驱动内部转换成
角度。双臂命名空间下右臂通常发布至 `/r_arm/rm_driver/JointPos`；实际话题应以机器人
电脑运行的 launch 和 `rostopic list` 为准。夹爪需要由机器人端映射到实际 AG2F90D
控制接口。

## 7. 必须实现的安全措施

服务端的 `0.05 rad` 限制不是硬件安全控制。机器人端仍必须实现：

- 按真实 RM65 的软限位裁剪或拒绝每个关节目标。
- 再次检查相邻目标不超过 `0.05 rad`。
- 网络、图像或关节状态超过 `0.5 s` 未更新时停止发送新动作。
- WebSocket 断开、响应超时、出现 NaN 或数组形状错误时保持当前位置并停止任务。
- 提供人工急停；首次测试使用低速度、无人员进入工作空间。
- 不执行 `request_id` 早于当前请求的过期响应。

建议先只记录服务端动作，不驱动机器人；确认关节顺序、方向和夹爪映射后，再逐步开放
单步执行和连续 5 步执行。

## 8. 通信测试

推理机或机器人端安装 `openpi-client` 后，可以发送全黑假图测试协议。该脚本不会连接
或控制机器人：

```bash
python3 client_smoke_test.py --host 10.147.17.223 --port 8000
```

仓库内示例位置：

```text
policy/pi05/examples/realman_remote/client_smoke_test.py
```
