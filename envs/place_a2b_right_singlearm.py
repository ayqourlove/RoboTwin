import glob
from ._base_task import Base_Task
from .utils import *
import sapien
import math
from ._GLOBAL_CONFIGS import *
import numpy as np


class place_a2b_right_singlearm(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):

        def get_available_model_ids(modelname):
            asset_path = os.path.join("assets/objects", modelname)
            json_files = glob.glob(os.path.join(asset_path, "model_data*.json"))

            available_ids = []
            for file in json_files:
                base = os.path.basename(file)
                try:
                    idx = int(base.replace("model_data", "").replace(".json", ""))
                    available_ids.append(idx)
                except ValueError:
                    continue
            return available_ids

        # 选择一批尺寸适中、适合单夹爪抓取的小物体。
        object_list = [
            "047_mouse",
            "048_stapler",
            "050_bell",
            "057_toycar",
            "073_rubikscube",
            "075_bread",
            "077_phone",
            "081_playingcards",
            "086_woodenblock",
            "112_tea-box",
            "113_coffee-box",
            "107_soap",
        ]
        object_list_np = np.array(object_list)

        try_num, try_lim = 0, 100
        while try_num <= try_lim:
            # A 是要被右手抓取并移动的物体，固定生成在桌面右侧。
            # 这里不要让 x 太靠中间，避免任务随机到左手更自然的位置。
            rand_pos = rand_pose(
                xlim=[0.24, 0.34],
                ylim=[-0.4, -0.3],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
            )

            # B 是参照物。它也放在右侧/中右侧，让真实单臂后续部署时仍在同一工作空间内。
            # 最终目标会在 B 的 +x 方向偏移 0.13m，即“放到 B 的右边”。
            target_rand_pose = rand_pose(
                xlim=[0.00, 0.10],
                ylim=[-0.2, 0.0],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
            )

            # 保证 A 和 B 初始位置不要太近，给专家轨迹留下明显的搬移动作。
            while (np.sqrt((target_rand_pose.p[0] - rand_pos.p[0])**2 + (target_rand_pose.p[1] - rand_pos.p[1])**2)
                   < 0.12) or (np.abs(target_rand_pose.p[1] - rand_pos.p[1]) < 0.08):
                target_rand_pose = rand_pose(
                    xlim=[0.00, 0.10],
                    ylim=[-0.2, 0.0],
                    qpos=[0.5, 0.5, 0.5, 0.5],
                    rotate_rand=True,
                    rotate_lim=[0, 3.14, 0],
                )
            try_num += 1

            target_place_x = target_rand_pose.p[0] + 0.13
            distance = np.sqrt(np.sum((rand_pos.p[:2] - target_rand_pose.p[:2])**2))

            # 目标点仍在右侧可达区域，同时 A 和 B 有足够间距时接受这组随机位置。
            if 0.13 <= target_place_x <= 0.25 and distance > 0.16:
                break

        if try_num > try_lim:
            raise "Actor create limit!"

        self.selected_modelname_A = np.random.choice(object_list_np)
        available_model_ids = get_available_model_ids(self.selected_modelname_A)
        if not available_model_ids:
            raise ValueError(f"No available model_data.json files found for {self.selected_modelname_A}")
        self.selected_model_id_A = np.random.choice(available_model_ids)

        self.object = create_actor(
            scene=self,
            pose=rand_pos,
            modelname=self.selected_modelname_A,
            convex=True,
            model_id=self.selected_model_id_A,
        )

        self.selected_modelname_B = np.random.choice(object_list_np)
        while self.selected_modelname_B == self.selected_modelname_A:
            self.selected_modelname_B = np.random.choice(object_list_np)

        available_model_ids = get_available_model_ids(self.selected_modelname_B)
        if not available_model_ids:
            raise ValueError(f"No available model_data.json files found for {self.selected_modelname_B}")

        self.selected_model_id_B = np.random.choice(available_model_ids)

        self.target_object = create_actor(
            scene=self,
            pose=target_rand_pose,
            modelname=self.selected_modelname_B,
            convex=True,
            model_id=self.selected_model_id_B,
        )

        self.object.set_mass(0.05)
        self.target_object.set_mass(0.05)
        self.add_prohibit_area(self.object, padding=0.05)
        self.add_prohibit_area(self.target_object, padding=0.1)

    def play_once(self):
        # 这个版本专门为单真实机械臂部署准备：即使仿真配置了左右臂，也始终只用右手。
        arm_tag = ArmTag("right")

        # 右手抓取 A。
        self.move(self.grasp_actor(self.object, arm_tag=arm_tag, pre_grasp_dis=0.1))

        # 抓起后沿机械臂坐标系抬高，避免平移时擦到桌面或参照物 B。
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))

        # 任务语义：把 A 放到 B 的右边，所以目标点取 B 的位置并沿 +x 偏移。
        target_pose = self.target_object.get_pose().p.tolist()
        target_pose[0] += 0.13

        # 放置完成后默认打开夹爪，后续成功条件也要求右手打开。
        self.move(self.place_actor(self.object, arm_tag=arm_tag, target_pose=target_pose))

        self.info["info"] = {
            "{A}": f"{self.selected_modelname_A}/base{self.selected_model_id_A}",
            "{B}": f"{self.selected_modelname_B}/base{self.selected_model_id_B}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        object_pose = self.object.get_pose().p
        target_pos = self.target_object.get_pose().p
        distance = np.sqrt(np.sum((object_pose[:2] - target_pos[:2])**2))

        # A 应该位于 B 的右侧，距离不能太近也不能太远，并且右手已经松开。
        return np.all(distance < 0.2 and distance > 0.08 and object_pose[0] > target_pos[0]
                      and abs(object_pose[1] - target_pos[1]) < 0.05 and self.robot.is_right_gripper_open())
