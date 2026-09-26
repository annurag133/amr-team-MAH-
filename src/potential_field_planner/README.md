# potential_field_planner

ROS 2 (Humble) package with the navigation, localisation and exploration nodes for the Robile.

## Nodes

### astar_global_planner

A* on the occupancy grid. Cells closer than `robot_radius` to an obstacle are blocked and cells near walls get an extra cost so paths stay in the middle of corridors. The path is thinned to waypoints about 0.4 m apart.

| | Topic | Type |
|---|---|---|
| sub | `/map` | nav_msgs/OccupancyGrid |
| sub | `/goal_pose` | geometry_msgs/PoseStamped |
| sub | `/replan` | std_msgs/Empty |
| sub | `/scan` | sensor_msgs/LaserScan (only used when replanning) |
| pub | `/global_path` | nav_msgs/Path |

Needs the `map -> base_link` transform.

### potential_field

Follows `/global_path`: attraction to the current waypoint, repulsion from laser points around the robot outline, a sideways component to get around obstacles, and a holonomic final approach for the last 0.5 m. Every command is checked for collisions over the next 0.5 s before it is sent.

| | Topic | Type |
|---|---|---|
| sub | `/global_path` | nav_msgs/Path |
| sub | `/scan` | sensor_msgs/LaserScan |
| sub | `/cancel_goal` | std_msgs/Empty |
| pub | `/cmd_vel` | geometry_msgs/Twist |
| pub | `/navigation_status` | std_msgs/String (IDLE, FOLLOWING, BLOCKED, ALIGNING, REACHED, FAILED, WAITING) |
| pub | `/replan` | std_msgs/Empty |

### mcl_localization

Particle filter with an odometry motion model, likelihood-field sensor model and low-variance resampling. Works as a replacement for AMCL.

| | Topic | Type |
|---|---|---|
| sub | `/map`, `/scan`, `/initialpose` | |
| pub | `/mcl_pose` | geometry_msgs/PoseWithCovarianceStamped |
| pub | `/particle_cloud` | nav2_msgs/ParticleCloud |
| pub | TF `map -> odom` | |
| srv | `/reinitialize_global_localization` | std_srvs/Empty |

### frontier_explorer

Finds frontiers on the SLAM map, picks the cheapest reachable one (travel distance minus a bonus for frontier length) and sends it to `/goal_pose`. Saves the map when nothing reachable is left.

| | Topic | Type |
|---|---|---|
| sub | `/map`, `/navigation_status` | |
| pub | `/goal_pose` | geometry_msgs/PoseStamped |
| pub | `/frontiers` | visualization_msgs/MarkerArray |
| pub | `/exploration_status` | std_msgs/String (EXPLORING, COMPLETE) |

## Launch arguments

`real_robot_nav.launch.py`

| Argument | Default | |
|---|---|---|
| `localization` | `mcl` | `mcl`, `amcl` or `none` |
| `map` | `maps/mapping_1.yaml` | map for localisation |
| `slam` | `false` | run slam_toolbox instead of map server + localisation |
| `explore` | `false` | start the frontier explorer (needs `slam:=true`) |
| `map_save_path` | `~/explored_map` | where the explored map is saved |
| `x`, `y`, `yaw` | `0` | initial pose guess |
| `rviz` | `true` | open RViz |
| `use_sim_time` | `false` | set to `true` in Gazebo |
