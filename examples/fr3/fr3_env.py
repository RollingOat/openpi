import franky
from realsense_camera_utils import start_realsense_pipeline, get_images_from_realsense
from web_camera_utils import get_images_from_web_camera, start_web_camera
from oak_camera_utils import start_multiple_oak_cameras, get_images_from_multiple_oak_cameras
import numpy as np

class robot_config:
    robot_ip: str = ""
    relative_dynamics_factor: float = 0.8
    action_space: str = "joint_velocity" # options: "joint_velocity", "end_effector_pose"
    gripper_action_space: str = "position" # options: "position", "open/close"
    control_frequency: int = 15


class camera_config:
    agent_view_camera_resolution: tuple = (640, 480)
    wrist_camera_resolution: tuple = (640, 480)
    frame_rate: int = 30
    use_agent_view_camera: bool = True
    use_wrist_camera: bool = True
    wrist_camera_type: str = "oak_camera"  # options: "realsense", "web_camera", "oak_camera"
    agent_view_camera_type: str = "oak_camera"  # options: "realsense", "web_camera", "oak_camera"
    wrist_oak_camera_id: str = "14442C10016FDAD600"
    agent_view_oak_camera_id: str = "14442C10117CC5D600"
    web_camera_index: int = 2  # default camera index for web camera

class FR3_ENV:
    def __init__(self, robot_config, camera_config):
        self.robot_config = robot_config
        self.camera_config = camera_config
        self.relative_max_joint_delta = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.max_joint_delta = self.relative_max_joint_delta.max()
        self.max_gripper_delta = 0.25
        self.max_lin_delta = 0.075
        self.max_rot_delta = 0.15
        self.control_hz = 15
        self._max_gripper_width = 0.079  # meters
        self.reset_joints = np.array([0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 1 / 4 * 3.1415])
        self.use_async_motion = True

        # if self.camera_config.use_agent_view_camera:
        #     # Initialize RealSense camera for agent view camera
        #     if self.camera_config.agent_view_camera_type == "realsense":
        #         self.agent_view_camera_pipeline = start_realsense_pipeline(
        #             rgb_resolution=self.camera_config.agent_view_camera_resolution,
        #             depth_resolution=self.camera_config.agent_view_camera_resolution,
        #             frame_rate=self.camera_config.frame_rate
        #         )
        #     elif self.camera_config.agent_view_camera_type == "web_camera":
        #         self.agent_view_camera_pipeline = start_web_camera(self.camera_config.web_camera_index)
        #     else:
        #         raise ValueError(f"Unsupported agent view camera type: {self.camera_config.agent_view_camera_type}")
        
        # if self.camera_config.use_wrist_camera:
        #     # Initialize RealSense camera for wrist camera
        #     if self.camera_config.wrist_camera_type == "realsense":
        #         self.wrist_camera_pipeline = start_realsense_pipeline(
        #             rgb_resolution=self.camera_config.wrist_camera_resolution,
        #             depth_resolution=self.camera_config.wrist_camera_resolution,
        #             frame_rate=self.camera_config.frame_rate
        #         )
        #     elif self.camera_config.wrist_camera_type == "web_camera":
        #         self.wrist_camera_pipeline = start_web_camera(self.camera_config.web_camera_index)
        #     else:
        #         raise ValueError(f"Unsupported wrist camera type: {self.camera_config.wrist_camera_type}")

        if self.camera_config.wrist_camera_type == "oak_camera" and self.camera_config.agent_view_camera_type == "oak_camera":
            self.oak_queues, self.oak_pipelines, self.oak_deviceInfos = start_multiple_oak_cameras()
        else:
            raise ValueError("Currently only oak_camera type is supported for both wrist and agent view cameras.")

        # connect to robot arm
        self.robot = franky.Robot(self.robot_config.robot_ip)
        self.robot.relative_dynamics_factor = self.robot_config.relative_dynamics_factor
        self.gripper = franky.Gripper(self.robot_config.robot_ip)
        # homing the gripper
        # print("Homing the gripper...")
        # self.gripper.homing()
        # print("Gripper homed.")

    def get_images(self):
        images = {}
        
        # # get wrist camera images
        # if self.camera_config.use_wrist_camera:
        #     wrist_image, _ = get_images_from_realsense(self.wrist_camera_pipeline)
        # else:
        #     wrist_image = None

        # # get agent view camera images
        # if self.camera_config.use_agent_view_camera:
        #     agent_view_image, _ = get_images_from_realsense(self.agent_view_camera_pipeline)
        # else:
        #     agent_view_image = None
        oak_images = get_images_from_multiple_oak_cameras(self.oak_queues, self.oak_deviceInfos)
        wrist_image = oak_images[self.camera_config.wrist_oak_camera_id]
        agent_view_image = oak_images[self.camera_config.agent_view_oak_camera_id]

        images["wrist_image"] = wrist_image
        images["agent_view_image"] = agent_view_image
        return images

    def get_observation(self):
        observation = {}
        
        # # get wrist camera images
        # if self.camera_config.use_wrist_camera:
        #     wrist_image, _ = get_images_from_realsense(self.wrist_camera_pipeline)
        # else:
        #     wrist_image = None

        # # get agent view camera images
        # if self.camera_config.use_agent_view_camera:
        #     agent_view_image, _ = get_images_from_realsense(self.agent_view_camera_pipeline)
        # else:
        #     agent_view_image = None
        wrist_image = self.get_images()["wrist_image"]
        agent_view_image = self.get_images()["agent_view_image"]

        # get joint positions
        joint_positions = self.get_robot_joint_positions()

        # get gripper position
        gripper_position = self.get_robot_gripper_position()

        # get cartisian pose
        cartisian_pose = self.get_robot_cartisian_pose()

        observation["wrist_image"] = wrist_image
        observation["agent_view_image"] = agent_view_image
        observation["joint_position"] = joint_positions
        observation["gripper_position"] = gripper_position
        observation["cartisian_pose"] = cartisian_pose
        return observation
    
    def step(self, action):
        action_dict = self.create_action_dict(action)
        print("Planned goal joint positions:", action_dict["joint_position"])
        print("Planned gripper position:", action_dict["gripper_position"])
        self.update_joints(action_dict["joint_position"])
        self.update_gripper(action_dict["gripper_position"])

        return action_dict
    
    def reset(self):
        print("resetting fr3 env...")
        self.update_gripper(0)
        print("gripper opened")
        self.update_joints(self.reset_joints)
        print("joints reset")


    def get_robot_joint_positions(self):
        joint_state = self.robot.current_joint_state
        return joint_state.position
    
    def get_robot_gripper_position(self):
        return self.gripper.width
    
    def get_robot_cartisian_pose(self):
        cartiesian_state = self.robot.current_cartesian_state
        robot_pose = cartiesian_state.pose
        ee_pose = robot_pose.end_effector_pose
        return ee_pose
    
    def create_action_dict(self, action):
        ## assuming action space is joint velocity + gripper position
        action_dict = {}
        action_dict["gripper_position"] = float(np.clip(action[-1], 0, 1))
        action_dict["joint_velocity"] = action[:-1]
        print("joint velocity is:", action_dict["joint_velocity"])
        joint_delta = self.joint_velocity_to_delta(action[:-1])
        print("joint delta is:", joint_delta)
        action_dict["joint_position"] = joint_delta + self.get_robot_joint_positions()
        print("goal joint position is:", action_dict["joint_position"])
        return action_dict
    

    def update_joints(self, joint_position):
        # If the robot entered a safety reflex (e.g. collision), the
        # Franka lib will reject move commands while in RobotMode.Reflex.
        # Try to recover before sending the motion command.
        try:
            robot_mode = self.robot.state.robot_mode
        except Exception:
            robot_mode = None

        if robot_mode == franky.RobotMode.Reflex:
            # recover_from_errors() returns a bool indicating success
            recovered = self.robot.recover_from_errors()
            if not recovered:
                raise RuntimeError("Robot is in Reflex mode and recover_from_errors() failed.")

        jpm = franky.JointMotion(joint_position)
        if self.use_async_motion:
            self.robot.move(jpm, asynchronous=True)
        else:
            self.robot.move(jpm)

    def update_gripper(self, gripper_position):
        command = float(np.clip(gripper_position, 0, 1))
        width = self._max_gripper_width * (1 - command)
        if not self.use_async_motion:
            self.gripper.move(width, speed=0.05)
        else:
            self.gripper.move_async(width, speed=0.05)

    ### Velocity To Delta ###
    def gripper_velocity_to_delta(self, gripper_velocity):
        gripper_vel_norm = np.linalg.norm(gripper_velocity)

        if gripper_vel_norm > 1:
            gripper_velocity = gripper_velocity / gripper_vel_norm

        gripper_delta = gripper_velocity * self.max_gripper_delta

        return gripper_delta

    def cartesian_velocity_to_delta(self, cartesian_velocity):
        if isinstance(cartesian_velocity, list):
            cartesian_velocity = np.array(cartesian_velocity)

        lin_vel, rot_vel = cartesian_velocity[:3], cartesian_velocity[3:6]

        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)

        if lin_vel_norm > 1:
            lin_vel = lin_vel / lin_vel_norm
        if rot_vel_norm > 1:
            rot_vel = rot_vel / rot_vel_norm

        lin_delta = lin_vel * self.max_lin_delta
        rot_delta = rot_vel * self.max_rot_delta

        return np.concatenate([lin_delta, rot_delta])

    def joint_velocity_to_delta(self, joint_velocity):
        if isinstance(joint_velocity, list):
            joint_velocity = np.array(joint_velocity)

        relative_max_joint_vel = self.joint_delta_to_velocity(self.relative_max_joint_delta)
        max_joint_vel_norm = (np.abs(joint_velocity) / relative_max_joint_vel).max()

        if max_joint_vel_norm > 1:
            joint_velocity = joint_velocity / max_joint_vel_norm

        joint_delta = joint_velocity * self.max_joint_delta

        return joint_delta

    ### Delta To Velocity ###
    def gripper_delta_to_velocity(self, gripper_delta):
        return gripper_delta / self.max_gripper_delta

    def cartesian_delta_to_velocity(self, cartesian_delta):
        if isinstance(cartesian_delta, list):
            cartesian_delta = np.array(cartesian_delta)

        cartesian_velocity = np.zeros_like(cartesian_delta)
        cartesian_velocity[:3] = cartesian_delta[:3] / self.max_lin_delta
        cartesian_velocity[3:6] = cartesian_delta[3:6] / self.max_rot_delta

        return cartesian_velocity

    def joint_delta_to_velocity(self, joint_delta):
        if isinstance(joint_delta, list):
            joint_delta = np.array(joint_delta)

        return joint_delta / self.max_joint_delta