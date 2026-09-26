from glob import glob
from setuptools import find_packages, setup

package_name = 'potential_field_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
        ('share/' + package_name + '/maps', glob('maps/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='moeed',
    maintainer_email='moied.awan10@gmail.com',
    description='A* + potential field navigation, MCL localisation and frontier exploration for the Robile',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'potential_field = potential_field_planner.potential_field:main',
            'occupancy_grid_mapper = potential_field_planner.occupancy_grid_mapper:main',
            'astar_global_planner = potential_field_planner.astar_global_planner:main',
            'mcl_localization = potential_field_planner.mcl_localization:main',
            'frontier_explorer = potential_field_planner.frontier_explorer:main',
        ],
    },
)
