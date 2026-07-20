from ._base_task import Base_Task
from .utils import *
import numpy as np
import sapien


class place_blue_block_right_green_singlearm(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.block_half_size = 0.025
        self.place_offset = 0.07

        # Blue block: fixed start pose, always manipulated by the right arm.
        blue_pose = sapien.Pose([0.0, -0.06, 0.783], [1, 0, 0, 0])
        self.blue_block = create_box(
            scene=self,
            pose=blue_pose,
            half_size=(self.block_half_size, self.block_half_size, self.block_half_size),
            color=(0, 0, 1),
            name="blue_block",
        )

        # Green block: randomized reference block. In RoboTwin tasks, "right side"
        # is usually represented as the +x direction on the table.
        green_pose = rand_pose(
            xlim=[0.1, 0.25],
            ylim=[-0.05, 0.05],
            zlim=[0.783],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )
        self.green_block = create_box(
            scene=self,
            pose=green_pose,
            half_size=(self.block_half_size, self.block_half_size, self.block_half_size),
            color=(0, 1, 0),
            name="green_block",
            is_static=True,
        )

        self.blue_block.set_mass(0.03)
        self.add_prohibit_area(self.blue_block, padding=0.07)
        self.add_prohibit_area(self.green_block, padding=0.07)

    def play_once(self):
        # Single-arm variant: even if the embodiment has two arms, only the right arm is used.
        arm_tag = ArmTag("right")

        # Pick up the fixed blue block.
        self.move(self.grasp_actor(self.blue_block, arm_tag=arm_tag, pre_grasp_dis=0.09, grasp_dis=0.01))

        # Lift before moving to reduce table and green-block collisions.
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08, move_axis="arm"))

        # Put the blue block center to the right side of the green block center, i.e. +x direction.
        green_pose = self.green_block.get_pose()
        target_pose = [
            green_pose.p[0] + self.place_offset,
            green_pose.p[1],
            green_pose.p[2],
            1,
            0,
            0,
            0,
        ]
        self.move(
            self.place_actor(
                self.blue_block,
                target_pose=target_pose,
                arm_tag=arm_tag,
                pre_dis=0.07,
                dis=0.0,
            ))

        # Move away after release so success is not blocked by gripper contact.
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.06, move_axis="arm"))

        self.info["info"] = {
            "{A}": "blue block",
            "{B}": "green block",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        blue_pose = self.blue_block.get_pose().p
        green_pose = self.green_block.get_pose().p
        dx = blue_pose[0] - green_pose[0]
        dy = abs(blue_pose[1] - green_pose[1])
        dz = abs(blue_pose[2] - green_pose[2])

        # Success: blue block center lies on the +x/right side of the green block,
        # roughly one block-width away, and the right gripper has released it.
        return (0.04 < dx < 0.11 and dy < 0.04 and dz < 0.02 and self.is_right_gripper_open())
