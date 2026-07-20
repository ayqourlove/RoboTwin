import argparse
import json
import socket
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from envs.robot.planner import CuroboPlanner


def parse_args():
    parser = argparse.ArgumentParser(description="cuRobo backend client for demo_sim_server.py.")
    parser.add_argument(
        "--embodiment",
        default=str(ROOT / "assets/embodiments/realman-new"),
        help="Embodiment folder containing config.yml and curobo_left/right.yml.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10087)
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    return parser.parse_args()


class SimplePose:
    def __init__(self, values):
        self.p = np.array(values[:3], dtype=np.float64)
        self.q = np.array(values[-4:], dtype=np.float64)

    def tolist(self):
        return self.p.tolist() + self.q.tolist()


class JsonLineClient:
    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port))
        self.reader = self.sock.makefile("r", encoding="utf-8")
        self.writer = self.sock.makefile("w", encoding="utf-8")

    def request(self, payload):
        self.writer.write(json.dumps(payload) + "\n")
        self.writer.flush()
        line = self.reader.readline()
        if not line:
            raise RuntimeError("simulator connection closed")
        response = json.loads(line)
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "unknown simulator error"))
        return response

    def close(self):
        self.sock.close()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_arm_index(arm):
    return 0 if arm == "left" else 1


def build_planner(embodiment_dir, cfg, arm, state):
    arm_idx = get_arm_index(arm)
    curobo_path = embodiment_dir / f"curobo_{arm}.yml"
    if not curobo_path.exists():
        curobo_path = embodiment_dir / "curobo.yml"

    return CuroboPlanner(
        SimplePose(state["root_pose"]),
        cfg["arm_joints_name"][arm_idx],
        state["joint_names"],
        yml_path=str(curobo_path),
    )


def print_help():
    print(
        "\nBackend commands:\n"
        "  pose x y z qw qx qy qz    Plan link6 to an absolute world pose.\n"
        "  offset dx dy dz            Plan link6 to current pose plus a world-frame offset.\n"
        "  current                    Ask simulator for current link6 pose and qpos.\n"
        "  arm left|right             Switch planning arm and rebuild planner.\n"
        "  reset                      Ask simulator to restore startup qpos.\n"
        "  shutdown                   Stop simulator server.\n"
        "  help                       Print this help.\n"
        "  quit                       Exit backend only.\n"
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
    if cmd == "shutdown":
        return "shutdown", []
    if cmd == "arm":
        return "arm", values
    if cmd in {"p", "pose", "target"}:
        return "pose", [float(x) for x in values]
    if cmd in {"o", "offset"}:
        return "offset", [float(x) for x in values]
    return "pose", [float(x) for x in parts]


def get_state(client, arm):
    return client.request({"cmd": "get_state", "arm": arm})["state"]


def print_state(state, arm):
    print(f"Current {arm}_link6_pose:", state["requested_link6_pose"])
    print("Current qpos:", state["qpos"])


def plan_and_send(client, planner, arm, target, state):
    qpos = np.array(state["qpos"], dtype=np.float32)
    if not np.all(np.isfinite(qpos)):
        print("[WARN] Simulator qpos is not finite. Try `reset` before planning.")
        print(qpos)
        return

    print(f"Planning {arm}_link_6 to world target pose: {target}")
    try:
        result = planner.plan_path(qpos, SimplePose(target), arms_tag=arm)
    except Exception as exc:
        print(f"[WARN] Planner raised an exception: {exc}")
        return
    print("Plan status:", result["status"])
    if result["status"] != "Success":
        print("[WARN] Planning failed. Simulator keeps running; enter another target or `reset`.")
        return

    print("Sending trajectory steps:", result["position"].shape[0])
    client.request({
        "cmd": "execute_trajectory",
        "arm": arm,
        "positions": result["position"].tolist(),
        "velocities": result["velocity"].tolist(),
    })
    print("Simulator execution done.")


def main():
    args = parse_args()
    embodiment_dir = Path(args.embodiment).resolve()
    cfg = load_yaml(embodiment_dir / "config.yml")

    client = JsonLineClient(args.host, args.port)
    arm = args.arm
    state = get_state(client, arm)
    planner = build_planner(embodiment_dir, cfg, arm, state)
    print(f"Connected to simulator at {args.host}:{args.port}")
    print_state(state, arm)
    print_help()

    while True:
        try:
            cmd, values = parse_interactive_command(input("backend> "))
        except ValueError as exc:
            print(f"[WARN] Cannot parse numbers: {exc}")
            continue
        except (EOFError, KeyboardInterrupt):
            print("\nExit backend.")
            client.close()
            return

        if cmd is None:
            continue
        if cmd == "quit":
            client.close()
            return
        if cmd == "help":
            print_help()
            continue
        if cmd == "current":
            state = get_state(client, arm)
            print_state(state, arm)
            continue
        if cmd == "reset":
            state = client.request({"cmd": "reset", "arm": arm})["state"]
            planner = build_planner(embodiment_dir, cfg, arm, state)
            print("Simulator reset done.")
            print_state(state, arm)
            continue
        if cmd == "shutdown":
            client.request({"cmd": "shutdown"})
            print("Simulator shutdown requested.")
            client.close()
            return
        if cmd == "arm":
            if len(values) != 1 or values[0] not in {"left", "right"}:
                print("[WARN] Usage: arm left|right")
                continue
            arm = values[0]
            state = get_state(client, arm)
            planner = build_planner(embodiment_dir, cfg, arm, state)
            print(f"Switched to {arm}.")
            print_state(state, arm)
            continue
        if cmd == "pose":
            if len(values) != 7:
                print("[WARN] `pose` needs 7 numbers: x y z qw qx qy qz")
                continue
            state = get_state(client, arm)
            plan_and_send(client, planner, arm, values, state)
            continue
        if cmd == "offset":
            if len(values) != 3:
                print("[WARN] `offset` needs 3 numbers: dx dy dz")
                continue
            state = get_state(client, arm)
            current = np.array(state["requested_link6_pose"], dtype=np.float64)
            target_p = current[:3] + np.array(values, dtype=np.float64)
            target = target_p.tolist() + current[-4:].tolist()
            plan_and_send(client, planner, arm, target, state)
            continue

        print("[WARN] Unknown command. Use `help` to print commands.")


if __name__ == "__main__":
    main()
