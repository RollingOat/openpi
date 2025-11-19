from fr3_env import FR3_ENV, camera_config, robot_config
import numpy as np
import time

if __name__ == "__main__":
    config_robot = robot_config()
    config_robot.robot_ip = "192.168.1.12"
    config_camera = camera_config()
    config_camera.use_camera = False  # set to True if RealSense cameras are connected
    env = FR3_ENV(config_robot, config_camera)

    # test getting robot states
    joint_positions = env.get_robot_joint_positions()
    print("Start Joint Positions:", joint_positions)
    print()
    # test getting gripper position
    gripper_position = env.get_robot_gripper_position()
    print("Start Gripper Position:", gripper_position)
    print()
    # test getting cartisian pose
    cartisian_pose = env.get_robot_cartisian_pose()
    print("Start Cartisian Pose:", cartisian_pose)
    print()
    # test control with dummy action -0.10536409 -0.08951445  0.11596467 -0.05605802  0.07249575 -0.06708133 0.14404739  0.90018606
    dummy_action1 = np.array([-0.10536409, -0.08951445,  0.11596467, -0.05605802,  0.07249575, -0.06708133, 0.14404739,  1.0])
    env.step(dummy_action1)
    print("Joint Positions after dummy action 1:", env.get_robot_joint_positions())
    print()

    # time.sleep(2)  # wait for 2 seconds

    # test control with another dummy action -0.07055885 -0.1478698   0.09153871 -0.11702921 -0.21245446  0.17992752 0.13308554  0.0124388
    dummy_action2 = np.array([-0.07055885, -0.1478698,   0.09153871, -0.11702921, -0.21245446,  0.17992752, 0.13308554,  0.0])
    env.step(dummy_action2)
    print("Joint Positions after dummy action 2:", env.get_robot_joint_positions())
    print()