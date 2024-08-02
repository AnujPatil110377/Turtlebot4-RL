#!/usr/bin/env python3
"""
Launch file for TurtleBot4 exploration environment.

This launch file sets up the complete simulation environment including
Gazebo world, robot spawning, and necessary ROS 2 nodes for RL training.

Author: Anuj Patil
Based on original work by anurye (https://github.com/anurye/gym-turtlebot)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    ExecuteProcess,
    TimerAction
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for exploration training."""
    
    # Package directories
    tb4_gz_sim_dir = get_package_share_directory('tb4_gz_sim')
    tb4_description_dir = get_package_share_directory('tb4_description')
    
    # Launch arguments
    world_name = LaunchConfiguration('world_name')
    robot_name = LaunchConfiguration('robot_name')
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    # Declare launch arguments
    declare_world_name = DeclareLaunchArgument(
        'world_name',
        default_value='static_world',
        description='Name of the Gazebo world to load'
    )
    
    declare_robot_name = DeclareLaunchArgument(
        'robot_name',
        default_value='turtlebot4',
        description='Name of the robot'
    )
    
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    # Gazebo simulation launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([tb4_gz_sim_dir, 'launch', 'gz_sim.launch.py'])
        ]),
        launch_arguments={
            'world_name': world_name,
            'use_sim_time': use_sim_time
        }.items()
    )
    
    # Robot description launch
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([tb4_description_dir, 'launch', 'robot_description.launch.py'])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )
    
    # Robot spawning
    spawn_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([tb4_gz_sim_dir, 'launch', 'spawn_tb4.launch.py'])
        ]),
        launch_arguments={
            'robot_name': robot_name,
            'x': '0.0',
            'y': '0.0',
            'z': '0.1',
            'yaw': '0.0'
        }.items()
    )
    
    # Dynamic obstacle manager (optional)
    obstacle_manager = Node(
        package='tb4_drl_navigation',
        executable='dynamic_obstacle_manager.py',
        name='obstacle_manager',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # RViz for visualization (optional)
    rviz_config = PathJoinSubstitution([
        tb4_description_dir, 'rviz', 'config_drl.rviz'
    ])
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    return LaunchDescription([
        # Arguments
        declare_world_name,
        declare_robot_name,
        declare_use_sim_time,
        
        # Simulation
        gazebo_launch,
        robot_description_launch,
        
        # Robot spawning with delay
        TimerAction(
            period=5.0,
            actions=[spawn_robot]
        ),
        
        # Additional nodes with delay
        TimerAction(
            period=8.0,
            actions=[obstacle_manager]
        ),
        
        # RViz with delay (optional)
        TimerAction(
            period=10.0,
            actions=[rviz_node]
        )
    ])