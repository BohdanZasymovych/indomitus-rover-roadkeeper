import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'rover_road_assist'

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bohdan',
    maintainer_email='bohdan.zasymovych@gmail.com',
    description="Camera-based road keeping for manual driving: segments the road, estimates the rover's offset and heading relative to it, and warns the operator on road departure or corrects the steering.",
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            f'road_assist_node = {package_name}.road_assist_node:main',
            f'road_deviation_node = {package_name}.road_deviation_node:main',
        ],
    },
)
