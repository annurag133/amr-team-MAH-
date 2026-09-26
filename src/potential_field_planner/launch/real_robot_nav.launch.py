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
    loc = {'true': 'amcl', 'false': 'none'}.get(arg('localization').lower(),
                                                arg('localization').lower())
    if loc not in ('amcl', 'none'):
        raise RuntimeError(f"localization must be amcl or none (got '{loc}')")
    x, y, yaw = float(arg('x')), float(arg('y')), float(arg('yaw'))

    out = []
    if loc == 'amcl':
        out.append(Node(package='nav2_map_server', executable='map_server',
                        name='map_server', output='screen',
                        parameters=[sim, {'yaml_filename': arg('map')}]))
        managed = ['map_server']
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
        out.append(Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                        name='lifecycle_manager_localization', output='screen',
                        parameters=[sim, {'autostart': True, 'bond_timeout': 0.0,
                                          'node_names': managed}]))

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
            'localization', default_value='amcl',
            description='amcl or none'),
        DeclareLaunchArgument('x', default_value='0.0', description='Robot start x in map'),
        DeclareLaunchArgument('y', default_value='0.0', description='Robot start y in map'),
        DeclareLaunchArgument('yaw', default_value='0.0', description='Robot start yaw in map'),
        OpaqueFunction(function=nodes),
    ])
