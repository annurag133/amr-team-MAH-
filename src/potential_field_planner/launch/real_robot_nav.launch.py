import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def nodes(context):
    nav_share = get_package_share_directory('robile_navigation')
    pkg_share = get_package_share_directory('potential_field_planner')

    arg = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    sim = {'use_sim_time': arg('use_sim_time').lower() == 'true'}
    params = arg('params_file')
    slam = arg('slam').lower() == 'true'
    loc = {'true': 'amcl', 'false': 'none'}.get(arg('localization').lower(),
                                                arg('localization').lower())
    if slam:
        loc = 'none'  # slam_toolbox provides both the map and map->odom
    if loc not in ('amcl', 'mcl', 'none'):
        raise RuntimeError(f"localization must be amcl, mcl or none (got '{loc}')")
    x, y, yaw = float(arg('x')), float(arg('y')), float(arg('yaw'))

    out = []
    if loc in ('amcl', 'mcl'):
        out.append(Node(package='nav2_map_server', executable='map_server',
                        name='map_server', output='screen',
                        parameters=[sim, {'yaml_filename': arg('map')}]))
        managed = ['map_server']
        if loc == 'amcl':
            out.append(Node(
                package='nav2_amcl', executable='amcl', name='amcl', output='screen',
                parameters=[os.path.join(nav_share, 'config', 'nav2_params.yaml'), sim, {
                    'set_initial_pose': True,
                    'alpha1': 0.2, 'alpha2': 0.2, 'alpha3': 0.2,
                    'alpha4': 0.2, 'alpha5': 0.2,
                    'max_beams': 60,
                    'update_min_d': 0.05, 'update_min_a': 0.05,
                    'laser_max_range': 8.0,
                    'initial_pose.x': x, 'initial_pose.y': y, 'initial_pose.yaw': yaw,
                }]))
            managed.append('amcl')
        else:
            out.append(Node(
                package='potential_field_planner', executable='mcl_localization',
                name='mcl_localization', output='screen',
                parameters=[params, sim, {
                    'initial_x': x, 'initial_y': y, 'initial_yaw': yaw}]))
        out.append(Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                        name='lifecycle_manager_localization', output='screen',
                        parameters=[sim, {'autostart': True, 'bond_timeout': 0.0,
                                          'node_names': managed}]))

    explore = arg('explore').lower() == 'true'
    if explore and not slam:
        raise RuntimeError('explore:=true needs slam:=true')
    if slam:
        out.append(Node(package='slam_toolbox', executable='async_slam_toolbox_node',
                        name='slam_toolbox', output='screen',
                        parameters=[os.path.join(nav_share, 'config',
                                                 'mapper_params_online_async.yaml'), sim,
                                    {'map_update_interval': 2.0,
                                     # add a scan every ~6 deg of turning / 10 cm of travel so
                                     # consecutive scans overlap enough for scan matching
                                     'minimum_travel_heading': 0.1,
                                     'minimum_travel_distance': 0.1}]))
    if explore:
        out.append(Node(package='potential_field_planner', executable='frontier_explorer',
                        name='frontier_explorer', output='screen',
                        parameters=[params, sim, {'map_save_path': arg('map_save_path')}]))

    out += [
        Node(package='potential_field_planner', executable='astar_global_planner',
             name='astar_global_planner', output='screen', parameters=[params, sim]),
        Node(package='potential_field_planner', executable='potential_field',
             name='potential_field_planner', output='screen', parameters=[params, sim]),
    ]
    if arg('rviz').lower() == 'true':
        out.append(Node(package='rviz2', executable='rviz2', name='rviz2', output='log',
                        arguments=['-d', os.path.join(pkg_share, 'rviz', 'task1.rviz')],
                        parameters=[sim]))
    return out


def generate_launch_description():
    pkg_share = get_package_share_directory('potential_field_planner')
    return LaunchDescription([
        DeclareLaunchArgument(
            'map', default_value=os.path.join(pkg_share, 'maps', 'mapping_1.yaml'),
            description='Full path to the map yaml'),
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(pkg_share, 'config', 'planner_params.yaml')),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument(
            'localization', default_value='mcl',
            description='amcl (nav2), mcl (our particle filter) or none'),
        DeclareLaunchArgument(
            'slam', default_value='false',
            description='Build the map live with slam_toolbox (overrides localization)'),
        DeclareLaunchArgument(
            'explore', default_value='false',
            description='Autonomous frontier exploration (use with slam:=true)'),
        DeclareLaunchArgument('map_save_path',
                              default_value=os.path.expanduser('~/explored_map')),
        DeclareLaunchArgument('x', default_value='0.0', description='Robot start x in map'),
        DeclareLaunchArgument('y', default_value='0.0', description='Robot start y in map'),
        DeclareLaunchArgument('yaw', default_value='0.0', description='Robot start yaw in map'),
        OpaqueFunction(function=nodes),
    ])
