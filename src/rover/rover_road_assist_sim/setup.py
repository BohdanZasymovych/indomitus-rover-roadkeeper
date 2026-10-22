import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'rover_road_assist_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'rviz'),
            glob(os.path.join('rviz', '*.rviz'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bohdan',
    maintainer_email='bohdan.zasymovych@gmail.com',
    description='Gazebo test bench for rover_road_assist: a generated road and fake '
                'road perception computed from the ground-truth pose.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            f'road_sim_node = {package_name}.road_sim_node:main',
        ],
    },
)
