import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from openpi_client import base_policy
import tyro

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as training_config

PI05_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    PI05_DIR / "checkpoints" / "pi05_base_adjust_bottle_singlearm_lora" / "adjust_bottle_singlearm_lora" / "10000"
)
DEFAULT_PROMPT = "Pick up the bottle and place it upright at the target position."
CAMERA_KEYS = ("middle", "left", "right")


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    checkpoint_dir: str = str(DEFAULT_CHECKPOINT)
    train_config_name: str = "pi05_base_adjust_bottle_singlearm_lora"
    prompt: str = DEFAULT_PROMPT
    action_horizon: int = 5
    max_joint_delta: float = 0.05
    image_height: int = 480
    image_width: int = 640
    warmup: bool = True


def discover_asset(checkpoint_dir: Path) -> tuple[str, Path]:
    assets_dir = checkpoint_dir / "assets"
    candidates = sorted(path for path in assets_dir.iterdir() if (path / "norm_stats.json").is_file())
    if len(candidates) != 1:
        names = [path.name for path in candidates]
        raise ValueError(f"Expected exactly one checkpoint asset with norm_stats.json, found: {names}")
    return candidates[0].name, candidates[0] / "norm_stats.json"


def load_state_mean(norm_stats_path: Path) -> np.ndarray:
    with norm_stats_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    stats = data.get("norm_stats", data)
    state_mean = np.asarray(stats["state"]["mean"], dtype=np.float32)
    if state_mean.shape != (14,) or not np.all(np.isfinite(state_mean)):
        raise ValueError(f"Expected a finite 14-D state mean in {norm_stats_path}, got {state_mean.shape}")
    return state_mean


class RealmanRemotePolicy(base_policy.BasePolicy):
    def __init__(
        self,
        policy: base_policy.BasePolicy,
        fixed_left_state: np.ndarray,
        *,
        prompt: str,
        action_horizon: int,
        max_joint_delta: float,
        image_height: int,
        image_width: int,
    ) -> None:
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if max_joint_delta <= 0:
            raise ValueError("max_joint_delta must be positive")

        self._policy = policy
        self._fixed_left_state = np.asarray(fixed_left_state, dtype=np.float32)
        if self._fixed_left_state.shape != (7,):
            raise ValueError(f"fixed_left_state must have shape (7,), got {self._fixed_left_state.shape}")

        self._prompt = prompt
        self._action_horizon = action_horizon
        self._max_joint_delta = max_joint_delta
        self._image_shape = (image_height, image_width, 3)

    def _prepare_image(self, image: Any, name: str) -> np.ndarray:
        array = np.asarray(image)
        if array.shape != self._image_shape:
            raise ValueError(f"images.{name} must have shape {self._image_shape}, got {array.shape}")
        if array.dtype != np.uint8:
            raise ValueError(f"images.{name} must use uint8 RGB pixels, got {array.dtype}")
        return np.ascontiguousarray(array.transpose(2, 0, 1))

    def _limit_actions(self, actions: np.ndarray, right_state: np.ndarray) -> np.ndarray:
        limited = np.asarray(actions[: self._action_horizon, 7:14], dtype=np.float32).copy()
        previous_joints = right_state[:6].copy()
        for index in range(limited.shape[0]):
            lower = previous_joints - self._max_joint_delta
            upper = previous_joints + self._max_joint_delta
            limited[index, :6] = np.clip(limited[index, :6], lower, upper)
            previous_joints = limited[index, :6]
        limited[:, 6] = np.clip(limited[:, 6], 0.0, 1.0)
        return limited

    def infer(self, obs: dict) -> dict:
        if not isinstance(obs, dict):
            raise ValueError("Request must be a dictionary")

        right_state = np.asarray(obs.get("right_state"), dtype=np.float32)
        if right_state.shape != (7,) or not np.all(np.isfinite(right_state)):
            raise ValueError("right_state must contain 7 finite values: 6 joint radians and one gripper value")
        if not 0.0 <= float(right_state[6]) <= 1.0:
            raise ValueError("right_state[6] must be in [0, 1], where 0 is closed and 1 is open")

        images = obs.get("images")
        if not isinstance(images, dict):
            raise ValueError("images must be a dictionary containing middle, left, and right")
        missing = set(CAMERA_KEYS) - set(images)
        if missing:
            raise ValueError(f"Missing camera images: {sorted(missing)}")

        model_obs = {
            "state": np.concatenate((self._fixed_left_state, right_state)),
            "images": {
                "cam_high": self._prepare_image(images["middle"], "middle"),
                "cam_left_wrist": self._prepare_image(images["left"], "left"),
                "cam_right_wrist": self._prepare_image(images["right"], "right"),
            },
            "prompt": self._prompt,
        }
        result = self._policy.infer(model_obs)
        full_actions = np.asarray(result.get("actions"))
        if full_actions.ndim != 2 or full_actions.shape[1] < 14:
            raise ValueError(f"Model returned invalid actions shape: {full_actions.shape}")
        if not np.all(np.isfinite(full_actions)):
            raise ValueError("Model returned non-finite actions")

        return {
            "actions": self._limit_actions(full_actions, right_state),
            "request_id": obs.get("request_id"),
            "raw_action_horizon": int(full_actions.shape[0]),
            "policy_timing": result.get("policy_timing", {}),
        }


def create_policy(args: Args) -> tuple[RealmanRemotePolicy, dict]:
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    asset_id, norm_stats_path = discover_asset(checkpoint_dir)
    state_mean = load_state_mean(norm_stats_path)

    logging.info("Loading checkpoint: %s", checkpoint_dir)
    trained_policy = policy_config.create_trained_policy(
        training_config.get_config(args.train_config_name),
        checkpoint_dir,
        robotwin_repo_id=asset_id,
    )
    remote_policy = RealmanRemotePolicy(
        trained_policy,
        state_mean[:7],
        prompt=args.prompt,
        action_horizon=args.action_horizon,
        max_joint_delta=args.max_joint_delta,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    metadata = {
        "protocol": "robotwin.realman.pi05.v1",
        "camera_order": list(CAMERA_KEYS),
        "image_shape": [args.image_height, args.image_width, 3],
        "image_dtype": "uint8",
        "image_color_order": "RGB",
        "right_state_order": [
            "right_joint_1",
            "right_joint_2",
            "right_joint_3",
            "right_joint_4",
            "right_joint_5",
            "right_joint_6",
            "right_gripper",
        ],
        "action_type": "absolute_joint_target",
        "action_horizon": args.action_horizon,
        "execution_hz": 10,
        "max_joint_delta_rad": args.max_joint_delta,
        "fixed_prompt": args.prompt,
        "example_right_state": state_mean[7:14],
    }
    return remote_policy, metadata


def warmup(policy: RealmanRemotePolicy, metadata: dict) -> None:
    logging.info("Warming up policy; the first JAX compilation may take a while...")
    height, width, _ = metadata["image_shape"]
    zero_image = np.zeros((height, width, 3), dtype=np.uint8)
    policy.infer(
        {
            "request_id": "warmup",
            "right_state": np.asarray(metadata["example_right_state"], dtype=np.float32),
            "images": {name: zero_image for name in CAMERA_KEYS},
        }
    )
    logging.info("Policy warmup complete")


def main(args: Args) -> None:
    policy, metadata = create_policy(args)
    if args.warmup:
        warmup(policy, metadata)

    logging.info("Serving RealMan pi05 on ws://%s:%d", args.host, args.port)
    logging.info("Health check: http://%s:%d/healthz", args.host, args.port)
    try:
        websocket_policy_server.WebsocketPolicyServer(
            policy=policy,
            host=args.host,
            port=args.port,
            metadata=metadata,
        ).serve_forever()
    except KeyboardInterrupt:
        logging.info("RealMan pi05 server stopped")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    main(tyro.cli(Args))
