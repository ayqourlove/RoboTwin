import argparse
import json
import queue
import socket
import threading
import time
from pathlib import Path

import numpy as np
import sapien.core as sapien
import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="SAPIEN simulator server for the cuRobo demo backend.")
    parser.add_argument(
        "--embodiment",
        default=str(ROOT / "assets/embodiments/realman-new"),
        help="Embodiment folder containing config.yml and urdf/.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10087)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--settle-steps", type=int, default=100)
    parser.add_argument(
        "--reset-settle-steps",
        type=int,
        default=0,
        help="Extra physics steps after reset. Default 0 keeps reset qpos exact.",
    )
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
            active_by_name[name].set_drive_target(float(value))
            active_by_name[name].set_drive_velocity_target(0.0)

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
        render_once(scene, viewer)


def render_once(scene, viewer):
    if viewer is not None:
        scene.update_render()
        viewer.render()


def get_arm_index(arm):
    return 0 if arm == "left" else 1


def get_link6_pose(robot, arm):
    joint = robot.find_joint_by_name(f"{arm}_joint_6")
    if joint is None:
        raise RuntimeError(f"Cannot find joint: {arm}_joint_6")
    return joint.child_link.get_entity_pose()


def pose_to_list(pose):
    return pose.p.tolist() + pose.q.tolist()


def execute_path(scene, viewer, robot, cfg, arm, positions, velocities):
    arm_idx = get_arm_index(arm)
    active_by_name = {joint.get_name(): joint for joint in robot.get_active_joints()}
    arm_joints = [active_by_name[name] for name in cfg["arm_joints_name"][arm_idx]]

    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    if positions.ndim != 2 or velocities.ndim != 2:
        raise ValueError("positions and velocities must be 2D arrays")
    if positions.shape != velocities.shape:
        raise ValueError(f"positions shape {positions.shape} != velocities shape {velocities.shape}")
    if positions.shape[1] != len(arm_joints):
        raise ValueError(f"trajectory dof {positions.shape[1]} != {arm} arm dof {len(arm_joints)}")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise ValueError("trajectory contains nan or inf")

    for qpos, qvel in zip(positions, velocities):
        qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        robot.set_qf(qf)
        for joint, target, velocity in zip(arm_joints, qpos, qvel):
            joint.set_drive_target(float(target))
            joint.set_drive_velocity_target(float(velocity))
        scene.step()
        render_once(scene, viewer)


def reset_robot(scene, viewer, robot, root_pose, initial_qpos, settle_steps):
    robot.set_root_pose(root_pose)
    robot.set_qpos(initial_qpos)
    if hasattr(robot, "set_qvel"):
        robot.set_qvel(np.zeros_like(initial_qpos))
    robot.set_qf(np.zeros_like(initial_qpos))
    set_drive_targets_from_qpos(robot)
    if settle_steps > 0:
        step_scene(scene, viewer, settle_steps)
    else:
        render_once(scene, viewer)


class SimulatorServer:
    def __init__(self, scene, viewer, robot, cfg, root_pose, initial_qpos, settle_steps):
        self.scene = scene
        self.viewer = viewer
        self.robot = robot
        self.cfg = cfg
        self.root_pose = root_pose
        self.initial_qpos = initial_qpos
        self.settle_steps = settle_steps
        self.requests = queue.Queue()
        self.running = True

    def start_socket_thread(self, host, port):
        thread = threading.Thread(target=self._socket_loop, args=(host, port), daemon=True)
        thread.start()
        return thread

    def _socket_loop(self, host, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((host, port))
            server.listen()
            print(f"Simulator server listening on {host}:{port}")
            while self.running:
                try:
                    conn, addr = server.accept()
                except OSError:
                    return
                threading.Thread(target=self._client_loop, args=(conn, addr), daemon=True).start()

    def _client_loop(self, conn, addr):
        with conn:
            reader = conn.makefile("r", encoding="utf-8")
            writer = conn.makefile("w", encoding="utf-8")
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                response_queue = queue.Queue(maxsize=1)
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as exc:
                    response = {"ok": False, "error": f"invalid json: {exc}"}
                else:
                    self.requests.put((request, response_queue))
                    response = response_queue.get()
                writer.write(json.dumps(response) + "\n")
                writer.flush()

    def run_main_loop(self):
        while self.running:
            if self.viewer is not None and self.viewer.closed:
                print("Viewer closed. Stop simulator server.")
                self.running = False
                break

            try:
                request, response_queue = self.requests.get(timeout=0.01)
            except queue.Empty:
                render_once(self.scene, self.viewer)
                time.sleep(0.005)
                continue

            try:
                response = self.handle_request(request)
            except Exception as exc:
                response = {"ok": False, "error": str(exc)}
            response_queue.put(response)

    def handle_request(self, request):
        cmd = request.get("cmd")
        if cmd == "get_state":
            return self.get_state(request.get("arm", "right"))
        if cmd == "execute_trajectory":
            execute_path(
                self.scene,
                self.viewer,
                self.robot,
                self.cfg,
                request["arm"],
                request["positions"],
                request["velocities"],
            )
            return {"ok": True, "state": self.get_state(request["arm"])["state"]}
        if cmd == "reset":
            reset_robot(
                self.scene,
                self.viewer,
                self.robot,
                self.root_pose,
                self.initial_qpos,
                self.settle_steps,
            )
            return {"ok": True, "state": self.get_state(request.get("arm", "right"))["state"]}
        if cmd == "shutdown":
            self.running = False
            return {"ok": True}
        raise ValueError(f"unknown cmd: {cmd}")

    def get_state(self, arm):
        qpos = self.robot.get_qpos()
        qvel = self.robot.get_qvel() if hasattr(self.robot, "get_qvel") else np.zeros_like(qpos)
        state = {
            "qpos": qpos.tolist(),
            "qvel": qvel.tolist(),
            "joint_names": [joint.get_name() for joint in self.robot.get_active_joints()],
            "root_pose": pose_to_list(self.root_pose),
            "left_link6_pose": pose_to_list(get_link6_pose(self.robot, "left")),
            "right_link6_pose": pose_to_list(get_link6_pose(self.robot, "right")),
            "requested_link6_pose": pose_to_list(get_link6_pose(self.robot, arm)),
        }
        return {"ok": True, "state": state}


def main():
    args = parse_args()
    embodiment_dir = Path(args.embodiment).resolve()
    cfg = load_yaml(embodiment_dir / "config.yml")

    scene, viewer = setup_scene(render=args.render)
    robot, root_pose = load_robot(scene, embodiment_dir, cfg)
    set_homestate(robot, cfg)
    step_scene(scene, viewer, args.settle_steps)

    initial_qpos = robot.get_qpos().copy()
    if not np.all(np.isfinite(initial_qpos)):
        raise RuntimeError(f"Robot qpos is not finite after startup: {initial_qpos}")

    server = SimulatorServer(scene, viewer, robot, cfg, root_pose, initial_qpos, args.reset_settle_steps)
    server.start_socket_thread(args.host, args.port)
    print("Initial left_link6_pose:", pose_to_list(get_link6_pose(robot, "left")))
    print("Initial right_link6_pose:", pose_to_list(get_link6_pose(robot, "right")))
    server.run_main_loop()


if __name__ == "__main__":
    main()
