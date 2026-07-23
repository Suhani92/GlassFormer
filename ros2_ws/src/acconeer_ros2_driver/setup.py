import os
from glob import glob
from setuptools import setup

package_name = 'acconeer_ros2_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.json')),
        (os.path.join('share', package_name), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Suhani Grover',
    maintainer_email='suhani1077@gmail.com',
    description='ROS2 driver for the Acconeer XM125 pulsed coherent radar and '
                'radar-guided glass mask generation for GlassFormer.',
    license='MIT',
    entry_points={
        'console_scripts': [
            # Radar driver: reads IQ from the XM125, publishes the range profile.
            'acconeer_iq_node = acconeer_ros2_driver.acconeer_iq_node:main',
            # Radar-guided mask generation (peak extraction, FoV projection,
            # depth-consistency filtering, morphology).
            'radar_mask_node = acconeer_ros2_driver.radar_mask_node:main',
            # Real-time GlassFormer segmentation inference.
            'glassformer_node = acconeer_ros2_driver.glassformer_node:main',
        ],
    },
)