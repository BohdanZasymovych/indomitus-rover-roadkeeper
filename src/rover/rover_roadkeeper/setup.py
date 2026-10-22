from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rover_roadkeeper'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test', 'scripts']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name] if os.path.exists('resource/' + package_name) else []),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='UCUSpaceRobotics',
    maintainer_email='indomitus@ucu.edu.ua',
    description='Few-Shot DINOv2 Road Segmentation and Autonomous Cruise Control',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'roadkeeper_node = rover_roadkeeper.ros.roadkeeper_node:main',
        ],
    },
)
