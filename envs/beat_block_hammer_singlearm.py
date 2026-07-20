from ._base_task import Base_Task
from .utils import *
import sapien
from ._GLOBAL_CONFIGS import *


class beat_block_hammer_singlearm(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.hammer = create_actor(
            scene=self,
            pose=sapien.Pose([0.0, -0.06, 0.783], [0, 0, 0.995, 0.105]),
            modelname="020_hammer",
            convex=True,
            model_id=0,
        )
        # block_pose = rand_pose(
        #     xlim=[0.1, 0.25],
        #     ylim=[-0.05, 0.05],
        #     zlim=[0.76],
        #     qpos=[1, 0, 0, 0],
        #     rotate_rand=True,
        #     rotate_lim=[0, 0, 0.5],
        # )
        # while abs(block_pose.p[0]) < 0.05 or np.sum(pow(block_pose.p[:2], 2)) < 0.001:
        #     block_pose = rand_pose(
        #         xlim=[0.1, 0.25],
        #         ylim=[-0.05, 0.05],
        #         zlim=[0.76],
        #         qpos=[1, 0, 0, 0],
        #         rotate_rand=True,
        #         rotate_lim=[0, 0, 0.5],
        #     )
        block_pose = sapien.Pose([0.1, -0.06, 0.783], [0, 0, 0.995, 0.105])
        self.block = create_box(
            scene=self,
            pose=block_pose,
            half_size=(0.025, 0.025, 0.025),
            color=(1, 0, 0),
            name="box",
            is_static=True,
        )
        self.hammer.set_mass(0.001)

        self.add_prohibit_area(self.hammer, padding=0.10)
        self.prohibited_area.append([
            block_pose.p[0] - 0.05,
            block_pose.p[1] - 0.05,
            block_pose.p[0] + 0.05,
            block_pose.p[1] + 0.05,
        ])

    def play_once(self):
        # Get the position of the block's functional point
        # block_pose = self.block.get_functional_point(0, "pose").p
        # Determine which arm to use based on block position (left if block is on left side, else right)
        arm_tag = ArmTag("right")

        # Grasp the hammer with the selected arm
        actions1 = self.grasp_actor(self.hammer, arm_tag=arm_tag, pre_grasp_dis=0.12, grasp_dis=0.01)
        self.move(actions1)
        # Move the hammer upwards
        actions2 = self.move_by_displacement(arm_tag, z=0.07, move_axis="arm")
        self.move(actions2)

        # Place the hammer on the block's functional point (position 1)
        
        actions3 = self.place_actor(
                self.hammer,
                target_pose=self.block.get_functional_point(1, "pose"),
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.06,
                dis=0,
                is_open=False,
                # pre_dis_axis="fp",
                # constrain="free"
            )
        # actions3[1][0].target_pose[2:] = actions2[1][0].target_pose[2:]
        # actions3[1][1].target_pose[2:] = actions2[1][0].target_pose[2:]

        self.move(actions3)
        
        self.info["info"] = {"{A}": "020_hammer/base0", "{a}": str(arm_tag)}
        return self.info

    # def play_once(self):

    #     # Get the position of the block's functional point
    #     # block_pose = self.block.get_functional_point(0, "pose").p
    #     # Determine which arm to use based on block position (left if block is on left side, else right)
    #     arm_tag = ArmTag("right")

    #     # Grasp the hammer with the selected arm
    #     actions = self.grasp_actor(self.hammer, arm_tag=arm_tag, pre_grasp_dis=0.12, grasp_dis=0.01)
    #     self.move(actions)
    #     action = actions[1][0]
    #     import transforms3d as t3d
    #     while True:
    #         right_ee_global_pose_q = list(self.robot.right_ee.global_pose.q)
    #         w_R_joint = t3d.quaternions.quat2mat(right_ee_global_pose_q)
    #         w_R_aloha = t3d.quaternions.quat2mat(action.target_pose[3:])
    #         ######## REMEMBER TO UPDATE THE DELTA_MATRIX!!!! ####
    #         # Update this delta_matrix with your calculated value
    #         delta_matrix = np.matrix([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    #         #####################################################
    #         global_trans_matrix = w_R_joint.T @ w_R_aloha @ delta_matrix.T
    #         print(np.round(global_trans_matrix))
    def check_success(self):
        hammer_target_pose = self.hammer.get_functional_point(0, "pose").p
        block_pose = self.block.get_functional_point(1, "pose").p
        eps = np.array([0.02, 0.02])
        return np.all(abs(hammer_target_pose[:2] - block_pose[:2]) < eps) and self.check_actors_contact(
            self.hammer.get_name(), self.block.get_name())
