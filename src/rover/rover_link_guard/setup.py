import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'rover_link_guard'

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
    description="Keeps the rover safe when the operator's link drops: detects the loss, then continues along the road or retreats along the recorded path until the link returns.",
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            f'link_guard_node = {package_name}.link_guard_node:main',
        ],
    },
)
