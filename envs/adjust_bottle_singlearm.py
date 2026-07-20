from ._base_task import Base_Task
from .utils import *


class adjust_bottle_singlearm(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # right_qpose = [0.707, 0.0, 0.0, -0.707]
        right_qpose = [1, 0.0, 0.0, 0]
        self.model_id = np.random.choice([13, 16])

        self.bottle = rand_create_actor(
            self,
            xlim=[-0.1, 0.1],
            ylim=[-0.10, -0.05],
            zlim=[0.752],
            rotate_rand=True,
            qpos=right_qpose,
            modelname="001_bottle",
            convex=True,
            rotate_lim=(0, 0, 3.14),
            model_id=self.model_id,
        )
        bottle_pose = self.bottle.get_pose()
        self.bottle_spawn_pose = bottle_pose.p.tolist() + bottle_pose.q.tolist()
        self.delay(4)
        self.add_prohibit_area(self.bottle, padding=0.15)

    def action_my(self, arm_tag=ArmTag("right")):
        bottle_center_pose = self.bottle.get_functional_point(0, "pose")
        orientation_matrix = np.asarray(self.bottle.config.get("orientation_point"), dtype=np.float64)
        if orientation_matrix.shape == (1, 4, 4):
            orientation_matrix = orientation_matrix[0]
        if bottle_center_pose is None or orientation_matrix.shape != (4, 4):
            raise ValueError("Bottle center or mouth point is not defined.")

        bottle_center = np.asarray(bottle_center_pose.p, dtype=np.float64)
        orientation_matrix = orientation_matrix.copy()
        orientation_matrix[:3, 3] *= np.asarray(self.bottle.config["scale"], dtype=np.float64)
        bottle_world_matrix = self.bottle.get_pose().to_transformation_matrix()
        bottle_mouth = (bottle_world_matrix @ orientation_matrix)[:3, 3]
        bottle_axis = bottle_mouth - bottle_center

        # Gripper +X points down. Its +Z follows the horizontal bottle axis.
        gripper_x = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        gripper_z = bottle_axis - np.dot(bottle_axis, gripper_x) * gripper_x
        gripper_z_norm = np.linalg.norm(gripper_z)
        if gripper_z_norm < 1e-8:
            raise ValueError("Bottle axis is parallel to world Z; grasp orientation is undefined.")
        gripper_z /= gripper_z_norm

        # gripper_z = -gripper_z

        gripper_y = np.cross(gripper_z, gripper_x)
        gripper_y /= np.linalg.norm(gripper_y)
        gripper_z = np.cross(gripper_x, gripper_y)
        gripper_z /= np.linalg.norm(gripper_z)

        gripper_rotation = np.column_stack((gripper_x, gripper_y, gripper_z))
        gripper_quat = t3d.quaternions.mat2quat(gripper_rotation).tolist()
        center_x, center_y = bottle_center[:2].tolist()

        action1_pose = [center_x, center_y, 1.0] + gripper_quat
        action2_pose = [center_x, center_y, 0.9] + gripper_quat
        action4_pose = [center_x, center_y, 1.1] + gripper_quat
        # action5_pose = [0.2967910942875937, -0.17212361821866634, 0.9469957605991061, 0.33363453956454353, -0.001198375642723288, -0.005011291879351232, 0.9426884134532864]
        action5_pose = [0.2, -0.1, 0.95, 0, 0, 0, 1]
        actions = [ 
            Action(arm_tag, "move", target_pose=action1_pose),
            Action(arm_tag, "move", target_pose=action2_pose),
            Action(arm_tag, "close", target_gripper_pos=0.0),
            Action(arm_tag, "move", target_pose=action4_pose),
            Action(arm_tag, "move", target_pose=action5_pose),
            Action(arm_tag, "open", target_gripper_pos=1.0),
        ]
        return arm_tag, actions

    def play_once(self):
        arm_tag = ArmTag("right")
        
        self.move(self.action_my(arm_tag))
        self.delay(25)

        self.info["info"] = {
            "{A}": f"001_bottle/base{self.model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        target_hight = 0.85
        bottle_pose = self.bottle.get_functional_point(0)
        return bottle_pose[2] > 0.85


# pre grasp pose: [0.09080805629491806, -0.1895170956850052, 0.9944212436676025, 0.1659744679927826, 0.6676888465881348, 0.23081311583518982, -0.6880184412002563]
# grasp pose: [0.08547924124138441, -0.1815335259348333, 0.8948829779394446, 0.1659744679927826, 0.6676888465881348, 0.23081311583518982, -0.6880184412002563]
# post grasp pose: [0.09069529171104777, -0.1890732510295291, 0.9952445245242535, 0.1650978449792185, 0.6676583714559484, 0.23200662746860456, -0.6878574891782837]
# place pose: [0.2967910942875937, -0.17212361821866634, 0.9469957605991061, 0.33363453956454353, -0.001198375642723288, -0.005011291879351232, 0.9426884134532864]
