import numpy as np
import robosuite as suite

print("all available environments: ", suite.ALL_ENVIRONMENTS)
print()
print("all available robots: ", suite.ALL_ROBOTS)
print()
print("all available controllers: ", suite.ALL_CONTROLLERS)
print()
print("all available grippers: ", suite.ALL_GRIPPERS)
print()



env_name = "Stack"  # try with other tasks like "Stack" and "Door"
robot_name = "Panda"  # try with other robots like "Sawyer" and

# create environment instance
env = suite.make(
    env_name=env_name, # try with other tasks like "Stack" and "Door"
    robots=robot_name,  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names="robot0_eye_in_hand_image",
)

# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    obs, reward, done, info = env.step(action)  # take action in the environment
    if i == 999:
        obs_dict = obs
        for key, value in obs_dict.items():
            print(f"{key}")
    
    # env.render()  # render on display