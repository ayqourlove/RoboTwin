import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sapien.core as sapien
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from envs.robot.planner import CuroboPlanner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal SAPIEN + cuRobo demo: move left/right link6 to a target world pose."
    )
    parser.add_argument(
        "--embodiment",
        default=str(ROOT / "assets/embodiments/realman-new"),
        help="Embodiment folder containing config.yml, urdf/, and curobo_left/right.yml.",
    )
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument(
        "--target",
        nargs=7,
        type=float,
        default=None,
        metavar=("x", "y", "z", "qw", "qx", "qy", "qz"),
        help="Absolute target pose of link6 in world frame.",
    )
    parser.add_argument(
        "--offset",
        nargs=3,
        type=float,
        default=None,
        metavar=("dx", "dy", "dz"),
        help="Optional first command: target = current link6 pose + this world-frame offset.",
    )
    parser.add_argument("--render", action="store_true", help="Open a SAPIEN viewer.")
    parser.add_argument("--settle-steps", type=int, default=100)
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_scene(render=False):
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene = engine.create_scene(sapien.SceneConfig())
    scene.set_timestep(1 / 250)
    scene.add_ground(0.0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 0.5, -1], [0.5, 0.5, 0.5], shadow=True)

    viewer = None
    if render:
        from sapien.utils.viewer import Viewer

        viewer = Viewer(renderer)
        viewer.set_scene(scene)
        viewer.set_camera_xyz(x=0.6, y=-0.8, z=1.4)
        viewer.set_camera_rpy(r=0.0, p=-0.7, y=2.4)

    return scene, viewer


def load_robot(scene, embodiment_dir, cfg):
    urdf_path = embodiment_dir / cfg["urdf_path"]
    root_pose_data = cfg.get("robot_pose", [[0, -0.65, 0.75, 1, 0, 0, 0]])[0]
    root_pose = sapien.Pose(root_pose_data[:3], root_pose_data[-4:])

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(str(urdf_path))
    robot.set_root_pose(root_pose)

    for joint in robot.get_active_joints():
        joint.set_drive_property(
            stiffness=cfg.get("joint_stiffness", 1000),
            damping=cfg.get("joint_damping", 200),
        )

    return robot, root_pose


def set_homestate(robot, cfg):
    active_by_name = {joint.get_name(): joint for joint in robot.get_active_joints()}
    for arm_idx, joint_names in enumerate(cfg["arm_joints_name"]):
        home = cfg.get("homestate", [[0] * len(joint_names), [0] * len(joint_names)])[arm_idx]
        for name, value in zip(joint_names, home):
            active_by_name[name].set_drive_target(value)
            active_by_name[name].set_drive_velocity_target(0.0)

    # Keep grippers in a simple fixed pose. The demo target is link6, not the gripper TCP.
    for gripper in cfg.get("gripper_name", []):
        base_joint = active_by_name.get(gripper["base"])
        if base_joint is not None:
            base_joint.set_drive_target(0.0)
            base_joint.set_drive_velocity_target(0.0)


def set_drive_targets_from_qpos(robot):
    qpos = robot.get_qpos()
    for joint, value in zip(robot.get_active_joints(), qpos):
        joint.set_drive_target(float(value))
        joint.set_drive_velocity_target(0.0)


def step_scene(scene, viewer, steps):
    for _ in range(steps):
        scene.step()
        if viewer is not None:
            scene.update_render()
            viewer.render()


def get_arm_index(arm):
    return 0 if arm == "left" else 1


def get_link6_pose(robot, arm):
    joint = robot.find_joint_by_name(f"{arm}_joint_6")
    if joint is None:
        raise RuntimeError(f"Cannot find joint: {arm}_joint_6")
    # cuRobo's ee_link is left_link_6/right_link_6, so use the child link pose.
    # joint.global_pose is the joint frame and can differ in orientation.
    return joint.child_link.get_entity_pose()


def build_planner(robot, root_pose, embodiment_dir, cfg, arm):
    arm_idx = get_arm_index(arm)
    curobo_path = embodiment_dir / f"curobo_{arm}.yml"
    if not curobo_path.exists():
        curobo_path = embodiment_dir / "curobo.yml"

    all_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    arm_joint_names = cfg["arm_joints_name"][arm_idx]
    return CuroboPlanner(
        root_pose,
        arm_joint_names,
        all_joint_names,
        yml_path=str(curobo_path),
    )


def execute_path(scene, viewer, robot, cfg, arm, plan_result):
    arm_idx = get_arm_index(arm)
    active_by_name = {joint.get_name(): joint for joint in robot.get_active_joints()}
    arm_joints = [active_by_name[name] for name in cfg["arm_joints_name"][arm_idx]]

    positions = plan_result["position"]
    velocities = plan_result["velocity"]
    for qpos, qvel in zip(positions, velocities):
        qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        robot.set_qf(qf)

        for joint, target, velocity in zip(arm_joints, qpos, qvel):
            joint.set_drive_target(float(target))
            joint.set_drive_velocity_target(float(velocity))

        scene.step()
        if viewer is not None:
            scene.update_render()
            viewer.render()


def reset_robot(scene, viewer, robot, root_pose, initial_qpos, settle_steps):
    robot.set_root_pose(root_pose)
    robot.set_qpos(initial_qpos)
    if hasattr(robot, "set_qvel"):
        robot.set_qvel(np.zeros_like(initial_qpos))
    if hasattr(robot, "set_qf"):
        robot.set_qf(np.zeros_like(initial_qpos))
    set_drive_targets_from_qpos(robot)
    step_scene(scene, viewer, settle_steps)


def pose_to_list(pose):
    return pose.p.tolist() + pose.q.tolist()


def print_help():
    print(
        "\nCommands:\n"
        "  pose x y z qw qx qy qz    Plan link6 to an absolute world pose.\n"
        "  offset dx dy dz            Plan link6 to current pose plus a world-frame offset.\n"
        "  current                    Print current link6 pose and robot qpos.\n"
        "  reset                      Restore the robot qpos captured after startup settling.\n"
        "  help                       Print this help.\n"
        "  quit                       Exit.\n"
        "\nYou can also enter 7 numbers directly as shorthand for pose.\n"
    )


def parse_interactive_command(line):
    parts = line.strip().split()
    if not parts:
        return None, []

    cmd = parts[0].lower()
    values = parts[1:]

    if cmd in {"q", "quit", "exit"}:
        return "quit", []
    if cmd in {"h", "help", "?"}:
        return "help", []
    if cmd in {"c", "current"}:
        return "current", []
    if cmd in {"r", "reset"}:
        return "reset", []
    if cmd in {"p", "pose", "target"}:
        return "pose", [float(x) for x in values]
    if cmd in {"o", "offset"}:
        return "offset", [float(x) for x in values]

    return "pose", [float(x) for x in parts]


def plan_and_execute(scene, viewer, robot, planner, cfg, arm, target):
    current_qpos = robot.get_qpos()
    if not np.all(np.isfinite(current_qpos)):
        print("[WARN] Robot qpos is not finite. Use `reset` before planning again.")
        print(current_qpos)
        return

    target_pose = sapien.Pose(target[:3], target[-4:])
    print(f"Planning {arm}_link_6 to world target pose: {target}")
    result = planner.plan_path(current_qpos, target_pose, arms_tag=arm)
    print("Plan status:", result["status"])
    if result["status"] != "Success":
        print("[WARN] Planning failed. The program is still running; enter another target or `reset`.")
        return

    print("Executing trajectory steps:", result["position"].shape[0])
    execute_path(scene, viewer, robot, cfg, arm, result)
    print("Done.")


def run_initial_command_if_requested(args, scene, viewer, robot, planner, cfg, arm):
    current_link6_pose = get_link6_pose(robot, arm)
    if args.target is not None:
        plan_and_execute(scene, viewer, robot, planner, cfg, arm, args.target)
    elif args.offset is not None:
        target_p = current_link6_pose.p + np.array(args.offset, dtype=np.float64)
        target = target_p.tolist() + current_link6_pose.q.tolist()
        plan_and_execute(scene, viewer, robot, planner, cfg, arm, target)


def interactive_loop(scene, viewer, robot, planner, cfg, arm, root_pose, initial_qpos, reset_settle_steps):
    print_help()
    while True:
        if viewer is not None and viewer.closed:
            print("Viewer closed. Exit.")
            return

        try:
            line = input("demo> ")
            cmd, values = parse_interactive_command(line)
        except ValueError as exc:
            print(f"[WARN] Cannot parse numbers: {exc}")
            continue
        except (EOFError, KeyboardInterrupt):
            print("\nExit.")
            return

        if cmd is None:
            continue
        if cmd == "quit":
            return
        if cmd == "help":
            print_help()
            continue
        if cmd == "current":
            current_pose = get_link6_pose(robot, arm)
            print(f"Current {arm}_link_6 pose:", pose_to_list(current_pose))
            print("Current qpos:", robot.get_qpos().tolist())
            continue
        if cmd == "reset":
            reset_robot(scene, viewer, robot, root_pose, initial_qpos, reset_settle_steps)
            print("Robot reset done.")
            continue
        if cmd == "pose":
            if len(values) != 7:
                print("[WARN] `pose` needs 7 numbers: x y z qw qx qy qz")
                continue
            plan_and_execute(scene, viewer, robot, planner, cfg, arm, values)
            continue
        if cmd == "offset":
            if len(values) != 3:
                print("[WARN] `offset` needs 3 numbers: dx dy dz")
                continue
            current_pose = get_link6_pose(robot, arm)
            target_p = current_pose.p + np.array(values, dtype=np.float64)
            target = target_p.tolist() + current_pose.q.tolist()
            plan_and_execute(scene, viewer, robot, planner, cfg, arm, target)
            continue

        print("[WARN] Unknown command. Use `help` to print commands.")


def main():
    args = parse_args()
    embodiment_dir = Path(args.embodiment).resolve()
    cfg = load_yaml(embodiment_dir / "config.yml")

    scene, viewer = setup_scene(render=args.render)
    robot, root_pose = load_robot(scene, embodiment_dir, cfg)
    set_homestate(robot, cfg)
    step_scene(scene, viewer, args.settle_steps)

    current_qpos = robot.get_qpos()
    if not np.all(np.isfinite(current_qpos)):
        raise RuntimeError(f"Robot qpos is not finite before planning: {current_qpos}")
    initial_qpos = current_qpos.copy()

    planner = build_planner(robot, root_pose, embodiment_dir, cfg, args.arm)
    current_link6_pose = get_link6_pose(robot, args.arm)
    print(
        f"Current {args.arm}_link_6 pose:",
        pose_to_list(current_link6_pose),
    )

    run_initial_command_if_requested(args, scene, viewer, robot, planner, cfg, args.arm)
    interactive_loop(
        scene,
        viewer,
        robot,
        planner,
        cfg,
        args.arm,
        root_pose,
        initial_qpos,
        reset_settle_steps=args.settle_steps,
    )


if __name__ == "__main__":
    main()
