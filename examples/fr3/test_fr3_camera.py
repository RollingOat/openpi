from fr3_env import FR3_ENV, camera_config, robot_config
import numpy as np
import cv2
import time

if __name__ == "__main__":
    config_robot = robot_config()
    config_robot.robot_ip = "192.168.2.12"
    config_camera = camera_config()
    use_wrist_camera = True
    use_agent_view_camera = True
    config_camera.use_wrist_camera = use_wrist_camera
    config_camera.use_agent_view_camera = use_agent_view_camera
    env = FR3_ENV(config_robot, config_camera)

    while True:
        start_time = time.time()
        # test get camera image
        observation = env.get_observation()
        wrist_image = observation["wrist_image"]
        agent_view_image = observation["agent_view_image"]

        # visualize the images using OpenCV
        if wrist_image is not None:
            cv2.namedWindow('Wrist Camera', cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Wrist Camera", wrist_image)
        if agent_view_image is not None:
            cv2.namedWindow('Agent View Camera', cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Agent View Camera", agent_view_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
            
        end_time = time.time()

        if end_time - start_time < 1/30.0:
            time.sleep(1/30.0 - (end_time - start_time))
    