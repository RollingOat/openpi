cmake_minimum_required(VERSION 3.5)
project(fr3)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate service from srv/Infer.srv and make sensor_msgs available to codegen
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/Pi05.srv"
  DEPENDENCIES sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)

# Install python service/node scripts
install(PROGRAMS
  scripts/policy_client.py
  scripts/infer_server.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()