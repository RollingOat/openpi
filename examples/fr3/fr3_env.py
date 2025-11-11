import franky
from realsense_camera_utils import start_realsense_pipeline, get_images_from_realsense

class robot_config:
    robot_ip: str = ""
    relative_dynamic_factor: float = 0.05
    action_space: str = "joint_velocity" # options: "joint_velocity", "end_effector_pose"
    gripper_action_space: str = "position" # options: "position", "open/close"
    control_frequency: int = 15


class camera_config:
    agent_view_camera_resolution: tuple = (640, 480)
    wrist_camera_resolution: tuple = (640, 480)
    frame_rate: int = 30

class FR3_ENV:
    def __init__(self, robot_config, camera_config):
        self.robot_config = robot_config
        self.camera_config = camera_config

        # Initialize RealSense camera for wrist camera
        self.wrist_camera_pipeline = start_realsense_pipeline(
            rgb_resolution=self.camera_config.wrist_camera_resolution,
            depth_resolution=self.camera_config.wrist_camera_resolution,
            frame_rate=self.camera_config.frame_rate
        )

        # connect to robot arm
        self.robot = franky.Robot(self.robot_config.robot_ip)
        self.robot.relative_dynamic_factor = self.robot_config.relative_dynamic_factor
        self.gripper = franky.Gripper(self.robot_config.robot_ip)

    def get_observation(self):
        observation = {}
        
        # get wrist camera images
        wrist_image, _ = get_images_from_realsense(self.wrist_camera_pipeline)
        # get agent view camera images
        agent_view_image, _ = get_images_from_realsense(self.agent_view_camera_pipeline)

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
        # action is a (8,) np.ndarray
        # joint velocity commands for 7 joints + gripper position command
        duration = franky.Duration(1.0 / self.robot_config.control_frequency * 1000)  # in milliseconds
        success = False
        if self.robot_config.action_space == "joint_velocity" and self.robot_config.gripper_action_space == "open/close":
            joint_velocity_command = action[:7]
            gripper_position_command = action[7]

            jv = franky.JointVelocityMotion(joint_velocity_command.tolist(), duration=duration)
            self.robot.move(jv, asynchronous=True)
            gripper_success = self.gripper.move_async(gripper_position_command, self.speed)
            success = gripper_success

        elif self.robot_config.action_space == "end_effector_pose" and self.robot_config.gripper_action_space == "open/close":
            raise NotImplementedError("End effector pose control not implemented yet.")
        
        else:
            raise ValueError("Unsupported action space configuration.")
        
        return success
    
    def unnormalize_action(self, action):
        # Assuming action is normalized between -1 and 1
        pass


    def get_robot_joint_positions(self):
        joint_state = self.robot.current_joint_states
        return joint_state.position
    
    def get_robot_gripper_position(self):
        return self.gripper.width
    
    def get_robot_cartisian_pose(self):
        cartiesian_state = self.robot.current_cartesian_states
        robot_pose = cartiesian_state.pose
        ee_pose = robot_pose.end_effector_pose
        return ee_pose
    
    