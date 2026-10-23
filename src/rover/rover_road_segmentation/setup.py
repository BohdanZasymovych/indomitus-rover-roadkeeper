import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'rover_road_segmentation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'weights'), glob(os.path.join('weights', '*.pt'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bohdan',
    maintainer_email='bohdan.zasymovych@gmail.com',
    description='Few-shot DINOv2 binary road segmentation publishing a mono8 0/1 road mask '
                'for rover_road_assist.',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            f'road_segmentation_node = {package_name}.segmentation_node:main',
        ],
    },
)
