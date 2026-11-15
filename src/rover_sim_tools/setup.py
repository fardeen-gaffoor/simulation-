from setuptools import setup

package_name = "rover_sim_tools"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Caterpillar Autonomy Challenge team",
    maintainer_email="team@cat-arena.local",
    description="Rover simulation tools: motor + battery electrical model.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "motor_battery_model = rover_sim_tools.motor_battery_model:main",
        ],
    },
)
