import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'rover_drone_detection'

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
        (os.path.join('share', package_name, 'models'),
            glob(os.path.join('models', '*.pt')) + glob(os.path.join('models', '*.md'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='andrii',
    maintainer_email='andriykulbabapetrovych@gmail.com',
    description='Camera-based detection of waiter FPV drones: warns the operator, draws the boxes '
                'and logs every sighting with the rover pose and the estimated drone position.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            f'drone_detection_node = {package_name}.drone_detection_node:main',
        ],
    },
)
