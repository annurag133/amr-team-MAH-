# AMR Project – Robile Navigation, Localisation and Exploration

Project for the Autonomous Mobile Robots course (SS26). Everything here runs on the real Robile and in simulation.

The project has three parts:

1. **Path and motion planning** – an A* global planner that produces waypoints, and a potential field controller that drives between them while avoiding obstacles.
2. **Localisation** – our own Monte Carlo localisation (particle filter), used instead of Nav2 AMCL.
3. **Exploration** – frontier-based exploration on top of slam_toolbox, so the robot maps an unknown room by itself.

The full write-up is in [docs/AMR_Project_Report.pdf](docs/AMR_Project_Report.pdf).

## Team

- Mohammad Moeed Ahsan – path and motion planning
- Anurag Tiwari – localisation
- Haider Qaizar Hussain – exploration and report

## Approach

**Path and motion planning.** A* runs on the occupancy grid with obstacles inflated by 0.30 m and an extra cost near walls, so paths stay in the middle of corridors. The path is thinned to waypoints ~0.4 m apart. The potential field follows them: unit attraction to the current waypoint, repulsion from laser points around the robot outline (capped below the attraction so it can steer but never cancel the goal), plus a sideways component that lets the robot slide around obstacles instead of getting stuck in front of them. For the last 0.5 m the Robile drives holonomically onto the goal. Every command is checked against the laser for collisions over the next half second before it is sent.

**Localisation.** Standard MCL from the lecture: 500 particles, odometry motion model with four noise parameters, likelihood-field sensor model on 60 beams, low-variance resampling when the effective particle count drops below half. It publishes `map -> odom` like AMCL, so the rest of the stack does not care which one runs. Global localisation spreads 3000 particles over the free space and only shrinks the set once they agree on one place.

**Exploration.** slam_toolbox builds the map. The explorer finds frontier cells (free next to unknown), groups them, and for each group looks for a reachable goal within 1 m that has enough room to turn. The cost is travel distance minus twice the frontier length, so bigger unexplored areas are preferred. Visited and failed goals are not picked again (failed ones get one retry at the end). When nothing reachable is left the map is saved.

## Repository layout

```
src/potential_field_planner/     ROS 2 package with all our nodes
  potential_field_planner/
    astar_global_planner.py      A* on the occupancy grid, publishes /global_path
    potential_field.py           waypoint follower (potential field), publishes /cmd_vel
    mcl_localization.py          particle filter, publishes map->odom TF
    frontier_explorer.py         picks frontier goals on the SLAM map
    scan_utils.py                laser -> robot frame projection (handles flipped lasers)
  launch/real_robot_nav.launch.py
  config/planner_params.yaml     all parameters in one place
  rviz/task1.rviz
  maps/mapping_1.*               map of the lab
docs/                            project report and figures
ros2_network_config.xml          FastDDS profile for talking to the robot over wifi
```

## Requirements

- Ubuntu 22.04, ROS 2 Humble
- `ros-humble-navigation2`, `ros-humble-nav2-bringup`, `ros-humble-slam-toolbox`
- Python: numpy, scipy
- The Robile packages from the course (`robile_navigation` etc.) in the same workspace

## Build

```bash
cd ~/ros2_ws/src
git clone https://github.com/annurag133/amr-team-MAH-.git
cp -r amr-team-MAH-/src/potential_field_planner .
cd ~/ros2_ws
colcon build --packages-select potential_field_planner
source install/setup.bash
```

## Running on the robot

On the robot (over ssh, ideally inside tmux):

```bash
ros2 launch robile_bringup robot.launch.py
```

On the laptop, connected to the Robile wifi:

```bash
export ROS_DOMAIN_ID=3        # our robot, check yours
source ~/ros2_ws/install/setup.bash
```

**Parts 1 and 2 – navigation on the saved map with our particle filter**

```bash
ros2 launch potential_field_planner real_robot_nav.launch.py
```

RViz opens. Set the robot pose with *2D Pose Estimate* (the laser should line up with the walls), then send a goal with *2D Goal Pose*. To compare with Nav2 AMCL add `localization:=amcl`, to use another map add `map:=/path/to/map.yaml`.

**Part 3 – autonomous exploration**

```bash
ros2 launch potential_field_planner real_robot_nav.launch.py slam:=true explore:=true
```

The robot does a slow turn, then drives from frontier to frontier (blue dots in RViz, the current target is the orange sphere). When it is done it prints `Exploration complete` and saves the map to `~/explored_map.yaml`. `ros2 topic echo /exploration_status` shows `EXPLORING` / `COMPLETE`.

To stop the robot at any time:

```bash
ros2 topic pub --once /cancel_goal std_msgs/msg/Empty
```

## Things we learned the hard way

- **Clock sync.** If the robot clock is off by even a few seconds, AMCL drops scans and nothing moves. We run chrony on the laptop and point the robot's timesyncd at it.
- **Laser mounting.** The laser frame on our Robile is flipped. All scan processing goes through the full 3D transform (`scan_utils.py`).
- **Turn slowly while mapping.** Fast in-place rotations make the odometry slip and slam_toolbox produced a doubled, rotated map. The explorer turns at 0.3 rad/s and the controller at most 0.5 rad/s.
- **RViz from the VS Code terminal** crashes if VS Code is a snap (libpthread symbol error). Use a normal terminal or unset the snap GTK variables.

## Parameters

All tunable values are in `config/planner_params.yaml`. The ones we changed most often:

| Parameter | Node | Meaning |
|---|---|---|
| `robot_radius` | A*, explorer | planning radius (0.30 m, lets the robot through ~0.7 m doors) |
| `k_rep`, `influence_dist` | potential field | how strongly / from how far obstacles push |
| `max_lin`, `max_ang` | potential field | speed limits |
| `num_particles` | MCL | particles while tracking (500) |
| `info_gain_weight` | explorer | preference for long frontiers over short trips |
