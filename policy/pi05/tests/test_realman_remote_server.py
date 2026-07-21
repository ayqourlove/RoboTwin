import numpy as np
from policy.pi05.realman_remote_server import RealmanRemotePolicy
import pytest


class FakePolicy:
    def __init__(self) -> None:
        self.last_obs = None

    def infer(self, obs: dict) -> dict:
        self.last_obs = obs
        actions = np.zeros((4, 14), dtype=np.float32)
        actions[:, 7:13] = 1.0
        actions[:, 13] = np.array([-1.0, 0.25, 0.75, 2.0], dtype=np.float32)
        return {"actions": actions, "policy_timing": {"infer_ms": 12.0}}


def make_request() -> dict:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    return {
        "request_id": 17,
        "right_state": np.zeros(7, dtype=np.float32),
        "images": {"middle": image, "left": image, "right": image},
    }


def test_maps_single_arm_request_and_limits_actions() -> None:
    inner = FakePolicy()
    policy = RealmanRemotePolicy(
        inner,
        np.arange(7, dtype=np.float32),
        prompt="fixed prompt",
        action_horizon=3,
        max_joint_delta=0.05,
        image_height=4,
        image_width=6,
    )

    response = policy.infer(make_request())

    np.testing.assert_array_equal(inner.last_obs["state"][:7], np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(inner.last_obs["state"][7:], np.zeros(7, dtype=np.float32))
    assert inner.last_obs["images"]["cam_high"].shape == (3, 4, 6)
    assert inner.last_obs["prompt"] == "fixed prompt"
    assert response["request_id"] == 17
    assert response["actions"].shape == (3, 7)
    np.testing.assert_allclose(response["actions"][:, 0], [0.05, 0.10, 0.15])
    np.testing.assert_allclose(response["actions"][:, 6], [0.0, 0.25, 0.75])


def test_rejects_invalid_image_shape() -> None:
    policy = RealmanRemotePolicy(
        FakePolicy(),
        np.zeros(7, dtype=np.float32),
        prompt="fixed prompt",
        action_horizon=3,
        max_joint_delta=0.05,
        image_height=4,
        image_width=6,
    )
    request = make_request()
    request["images"]["middle"] = np.zeros((4, 6), dtype=np.uint8)

    with pytest.raises(ValueError, match="images.middle must have shape"):
        policy.infer(request)
