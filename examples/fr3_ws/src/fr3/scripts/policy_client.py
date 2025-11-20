#!/usr/bin/env python3
"""ROS2 helper to request policy inference from a policy server.

This script implements a ROS service client: it serializes an observation with
`openpi_client.msgpack_numpy`, places the bytes (or a base64 string) into a
service request field (e.g. `observation`/`data`), calls the service, and
unpacks the returned action.

Usage example:
    python3 main_ros.py --service-type my_pkg.srv.Infer --service-name /infer_policy

"""

# import logging

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
        
import contextlib
import dataclasses
import faulthandler
import signal
import time
# from moviepy.editor import ImageSequenceClip
import numpy as np
# from openpi_client import image_tools
# from openpi_client import websocket_client_policy
# import pandas as pd
# from PIL import Image
from fr3_env import FR3_ENV, camera_config, robot_config
from fr3.srv import Pi05
import tqdm

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    # Hardware parameters
    # left_camera_id: str = "<your_camera_id>"  # e.g., "24259877"
    # right_camera_id: str = "<your_camera_id>"  # e.g., "24514023"
    # wrist_camera_id: str = "<your_camera_id>"  # e.g., "13062452"

    # Policy parameters
    # external_camera: str | None = (
    #     None  # which external camera should be fed to the policy, choose from ["left", "right"]
    # )

    # Rollout parameters
    max_timesteps: int = 600
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    # Remote server parameters
    # remote_host: str = "10.125.145.18"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    # remote_port: int = (
    #     8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    # )


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


class PolicyRequester(Node):
    def __init__(
        self,
        node_name: str = "policy_requester"
    ) -> None:
        super().__init__(node_name)
        self._service_client = self.create_client(Pi05, "infer_action")
        self.req = Pi05.Request()
        self.bridge = CvBridge()


    def infer(self, obs: dict, timeout: float = 5.0):
        wrist_image = obs["wrist_image"]
        agent_view_image = obs["agent_view_image"]
        joint_position = obs["joint_position"]
        gripper_position = obs["gripper_position"]
        prompt = obs["prompt"]

        # convert images in numpy to sensor_msgs/Image
        self.req.wrist_image = self.bridge.cv2_to_imgmsg(wrist_image, encoding="rgb8")
        self.req.agent_view_image = self.bridge.cv2_to_imgmsg(agent_view_image, encoding="rgb8")
        self.req.joint_position = joint_position
        self.req.gripper_position = gripper_position
        self.req.prompt = prompt

        future = self._service_client.call_async(self.req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if future.done():
            resp = future.result()
            # convert response action list to numpy array
            actions = np.array(resp.action_chunk).reshape(-1, 8)
            
        else:
            raise TimeoutError("Timed out waiting for service response")
        return actions


def main(args: Args | None = None):
    if args is None:
        args = Args()
    rclpy.init()
    robot_config_instance = robot_config()
    robot_config_instance.robot_ip = "192.168.2.12"
    robot_config_instance.action_space = "joint_velocity"
    robot_config_instance.gripper_action_space = "position"
    camera_config_instance = camera_config()
    camera_config_instance.agent_view_camera_resolution = (640, 480)
    camera_config_instance.wrist_camera_resolution = (640, 480)
    camera_config_instance.frame_rate = 30
    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    # FR3_ENV currently accepts (robot_config, camera_config)
    dummy_mode= True
    env = FR3_ENV(robot_config_instance, camera_config_instance, dummy_mode=dummy_mode)
    print("Created the fr3 env!")
    policy_client = PolicyRequester("policy_requester")
    print("reset the env!")
    env.reset()
    while rclpy.ok():
        
        instruction = input("Enter instruction: ")

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Prepare to save video of rollout
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        for t_step in bar:
            start_time = time.time()
            try:
                curr_obs = env.get_observation()

                # Send request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                    # and improve latency.
                    # request_data = {
                    #     "observation/exterior_image_1_left": image_tools.resize_with_pad(
                    #         curr_obs["agent_view_image"], 224, 224
                    #     ),
                    #     "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                    #     "observation/joint_position": curr_obs["joint_position"],
                    #     "observation/gripper_position": curr_obs["gripper_position"],
                    #     "prompt": instruction,
                    # }

                    request_data = {
                        "wrist_image": curr_obs["wrist_image"],
                        "agent_view_image": curr_obs["agent_view_image"],
                        "joint_position": curr_obs["joint_position"],
                        "gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                    }


                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    # with prevent_keyboard_interrupt():
                    # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
                    # infer() returns a numpy array shaped (10, 8)
                    pred_action_chunk = policy_client.infer(request_data)
                    rclpy.spin_once(policy_client, timeout_sec=0)
                    assert pred_action_chunk.shape == (10, 8)

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Binarize gripper action
                # 1 for closed, 0 for open
                if action[-1].item() > 0.5:
                    # action[-1] = 1.0
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    # action[-1] = 0.0
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                # clip all dimensions of action to [-1, 1]
                action = np.clip(action, -1, 1)
                print("received action:", action)

                env.step(action)

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / CONTROL_FREQUENCY:
                    time.sleep(1 / CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        # if input("Do one more eval? (enter y or n) ").lower() != "y":
        #     break

    env.reset()

if __name__ == "__main__":
    main()
