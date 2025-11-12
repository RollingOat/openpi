from fr3_env import FR3_ENV, camera_config, robot_config
import numpy as np
import cv2

if __name__ == "__main__":
    env = FR3_ENV(robot_config(), camera_config())

    # test get camera image
    observation = env.get_observation()
    wrist_image = observation["wrist_image"]
    agent_view_image = observation["agent_view_image"]

    print("Wrist Image Shape:", wrist_image.shape)
    print("Agent View Image Shape:", agent_view_image.shape)

    # visualize the images using OpenCV
    
    cv2.imshow("Wrist Camera", wrist_image)
    cv2.imshow("Agent View Camera", agent_view_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()