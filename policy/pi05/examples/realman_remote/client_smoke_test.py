import argparse
import logging

import numpy as np
from openpi_client import websocket_client_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one synthetic request without controlling a robot.")
    parser.add_argument("--host", default="10.147.17.223")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = client.get_server_metadata()
    print("Server metadata:", metadata)

    height, width, _ = metadata["image_shape"]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    request = {
        "request_id": "smoke-test-1",
        "right_state": np.asarray(metadata["example_right_state"], dtype=np.float32),
        "images": {
            "middle": image,
            "left": image,
            "right": image,
        },
    }
    response = client.infer(request)
    print("request_id:", response["request_id"])
    print("actions shape:", response["actions"].shape)
    print("actions:\n", response["actions"])
    print("timing:", response.get("server_timing"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
