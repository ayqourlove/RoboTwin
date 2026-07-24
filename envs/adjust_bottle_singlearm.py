from ._base_task import Base_Task
from .utils import *


class adjust_bottle_singlearm(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)
        self.info["info"] = {
            "{A}": f"001_bottle/base{self.model_id}",
            "{a}": str(ArmTag("right")),
        }

    def load_actors(self):
        # right_qpose = [0.707, 0.0, 0.0, -0.707]
        right_qpose = [1, 0.0, 0.0, 0]
        self.model_id = np.random.choice([13, 16])
        self.place_ee_pose = [0.2, -0.1, 0.95, 0, 0, 0, 1]
        # Expected bottle center after executing place_ee_pose with the current grasp offset.
        self.bottle_target_xy = np.array([0.055, -0.1], dtype=np.float64)

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
        action5_pose = self.place_ee_pose.copy()
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

        return self.info

    def check_success(self):
        bottle_center = np.asarray(self.bottle.get_functional_point(0)[:3], dtype=np.float64)

        orientation_matrix = np.asarray(self.bottle.config.get("orientation_point"), dtype=np.float64)
        if orientation_matrix.shape == (1, 4, 4):
            orientation_matrix = orientation_matrix[0]
        if orientation_matrix.shape != (4, 4):
            self.last_success_check = {
                "success": False,
                "reason": "invalid_orientation_point",
                "orientation_point_shape": list(orientation_matrix.shape),
            }
            return False

        orientation_matrix = orientation_matrix.copy()
        orientation_matrix[:3, 3] *= np.asarray(self.bottle.config["scale"], dtype=np.float64)
        bottle_world_matrix = self.bottle.get_pose().to_transformation_matrix()
        bottle_mouth = (bottle_world_matrix @ orientation_matrix)[:3, 3]
        bottle_axis = bottle_mouth - bottle_center
        bottle_half_height = np.linalg.norm(bottle_axis)
        if bottle_half_height < 1e-8:
            self.last_success_check = {
                "success": False,
                "reason": "invalid_bottle_axis",
                "bottle_center": bottle_center.tolist(),
                "bottle_mouth": bottle_mouth.tolist(),
            }
            return False
        bottle_axis /= bottle_half_height

        linear_velocity = None
        angular_velocity = None
        for component in self.bottle.actor.get_components():
            if hasattr(component, "get_linear_velocity"):
                linear_velocity = np.asarray(component.get_linear_velocity(), dtype=np.float64)
                angular_velocity = np.asarray(component.get_angular_velocity(), dtype=np.float64)
                break
        if linear_velocity is None or angular_velocity is None:
            self.last_success_check = {
                "success": False,
                "reason": "velocity_component_not_found",
            }
            return False

        target_table_height = 0.74 + self.table_z_bias + bottle_half_height
        xy_error = np.linalg.norm(bottle_center[:2] - self.bottle_target_xy)
        table_height_error = abs(bottle_center[2] - target_table_height)
        upright_cosine = np.dot(bottle_axis, np.array([0.0, 0.0, 1.0]))
        linear_speed = np.linalg.norm(linear_velocity)
        angular_speed = np.linalg.norm(angular_velocity)

        position_ok = xy_error < 0.05
        on_table = table_height_error < 0.035
        upright = upright_cosine > np.cos(np.deg2rad(20.0))
        stable = linear_speed < 0.02 and angular_speed < 0.2
        released = self.is_right_gripper_open()

        success = bool(position_ok and on_table and upright and stable and released)
        self.last_success_check = {
            "success": success,
            "position_ok": bool(position_ok),
            "on_table": bool(on_table),
            "upright": bool(upright),
            "stable": bool(stable),
            "released": bool(released),
            "bottle_center": bottle_center.tolist(),
            "target_xy": self.bottle_target_xy.tolist(),
            "xy_error": float(xy_error),
            "target_center_z": float(target_table_height),
            "table_height_error": float(table_height_error),
            "bottle_axis": bottle_axis.tolist(),
            "upright_cosine": float(upright_cosine),
            "linear_velocity": linear_velocity.tolist(),
            "linear_speed": float(linear_speed),
            "angular_velocity": angular_velocity.tolist(),
            "angular_speed": float(angular_speed),
        }
        return success


# pre grasp pose: [0.09080805629491806, -0.1895170956850052, 0.9944212436676025, 0.1659744679927826, 0.6676888465881348, 0.23081311583518982, -0.6880184412002563]
# grasp pose: [0.08547924124138441, -0.1815335259348333, 0.8948829779394446, 0.1659744679927826, 0.6676888465881348, 0.23081311583518982, -0.6880184412002563]
# post grasp pose: [0.09069529171104777, -0.1890732510295291, 0.9952445245242535, 0.1650978449792185, 0.6676583714559484, 0.23200662746860456, -0.6878574891782837]
# place pose: [0.2967910942875937, -0.17212361821866634, 0.9469957605991061, 0.33363453956454353, -0.001198375642723288, -0.005011291879351232, 0.9426884134532864]
