#!/usr/bin/env python3
"""ROS2 service server that accepts packed observations and returns inferred actions.

This server uses `example_interfaces.srv.String` for simplicity: the request's
`data` field should contain a base64-encoded msgpack byte payload (or raw bytes,
which will also be handled). The server unpacks the observation, runs a policy
inference (if a trained policy is provided) or an echo dummy policy, and returns
the action packed with msgpack and base64-encoded in the response `data` field.

Usage:
  # Run a dummy echo server
  python3 service_infer_server.py --service-name /infer_policy

  # Run a trained policy server (provide config and checkpoint-dir)
  python3 service_infer_server.py --service-name /infer_policy --config pi0_aloha_sim --checkpoint-dir gs://openpi-assets/checkpoints/pi0_aloha_sim

"""
import argparse
import logging
import rclpy
from rclpy.node import Node
# from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
# from openpi.shared import download
from openpi.training import config as _config
import numpy as np
# import cv_bridge
from cv_bridge import CvBridge
from fr3.srv import Pi05


def load_trained_policy(config_name: str, checkpoint_dir: str):
    config = _config.get_config(config_name)
    # checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")

    # Create a trained policy.
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    # # Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
    # example = droid_policy.make_droid_example()
    # result = policy.infer(example)

    return policy

class dummyPolicy:
    def __init__(self) -> None:
        pass

    def infer(self, obs):
        # Echo the joint_position (if present) and set a no-op gripper action
        # random 10 by 8 action chunk
        action = np.random.rand(10, 8)
        return action


class PolicyServiceServer(Node):
    def __init__(self, service_name: str, policy) -> None:
        super().__init__("policy_infer_service")
        self._policy = policy
        self._srv = self.create_service(Pi05, service_name, self._handle_request)
        self.cv_bridge = CvBridge()

    def _handle_request(self, req, resp):
        print("Received inference request.")
        wrist_image = req.wrist_image
        agent_view_image = req.agent_view_image
        joint_position = req.joint_position
        gripper_position = req.gripper_position
        prompt = req.prompt

        # convert sensor_msgs/Image to numpy
        wrist_image = self.cv_bridge.imgmsg_to_cv2(wrist_image, desired_encoding="rgb8")
        agent_view_image = self.cv_bridge.imgmsg_to_cv2(agent_view_image, desired_encoding="rgb8")

        obs = {
            "observation/exterior_image_1_left": agent_view_image,
            "observation/wrist_image_left": wrist_image,
            "observation/joint_position": joint_position,
            "observation/gripper_position": gripper_position,
            "prompt": prompt,
        }


        action_chunk = self._policy.infer(obs)
        resp.action_chunk = action_chunk.flatten().tolist()
        resp.success = True
        if resp.success:
            print("Inference successful, returning action chunk.")
        else:
            print("Inference failed.")
        return resp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-name", default="infer_action")
    parser.add_argument("--config", dest="config_name", default=None, help="Training config name (e.g., pi0_aloha_sim)")
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir", default=None, help="Checkpoint dir or gs:// path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    rclpy.init()
    
    if args.config_name and args.checkpoint_dir:
        policy = load_trained_policy(args.config_name, args.checkpoint_dir)
        print("Loaded trained policy.")
    else:
        policy = dummyPolicy()
        print("Using dummy echo policy.")
    
    server = PolicyServiceServer(args.service_name, policy)
    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        pass
    finally:
        server.destroy_node()


if __name__ == "__main__":
    main()
