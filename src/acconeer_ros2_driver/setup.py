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
    maintainer='suhani',
    maintainer_email='suhani1077@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 'acconeer_node = acconeer_ros2_driver.acconeer_node:main',
            'acconeer_iq_node = acconeer_ros2_driver.acconeer_iq_node:main', 
            # 'acconeer_raw_node = acconeer_ros2_driver.acconeer_raw_node:main',
            'glass_basic.py = acconeer_ros2_driver.glass_basic:main',
            # 'glass_detector.py = acconeer_ros2_driver.glass_detector:main',
            # 'glass_viz.py = acconeer_ros2_driver.glass_viz:main',
            # 'glass_adaptive.py = acconeer_ros2_driver.glass_adaptive:main',
            # 'glass_geometric.py = acconeer_ros2_driver.glass_geometric:main',
            # 'glass_grid.py = acconeer_ros2_driver.glass_grid:main',
            # 'glass_morph.py = acconeer_ros2_driver.glass_morph:main',
            # 'glass_slic.py = acconeer_ros2_driver.glass_slic:main',
            # 'glass_mincc.py = acconeer_ros2_driver.glass_mincc:main',
            # 'glass_ir_rf.py = acconeer_ros2_driver.glass_ir_rf:main',
            # 'glass_multipeak.py = acconeer_ros2_driver.glass_multipeak:main',
            # 'glass_depth.py = acconeer_ros2_driver.glass_depth:main',
        ],
    },
)