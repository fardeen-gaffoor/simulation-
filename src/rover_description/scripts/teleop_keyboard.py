#!/usr/bin/env python3
"""Keyboard teleop: drive the rover with WASD keys.

W = forward, S = backward, A = turn left, D = turn right
X or SPACE = stop driving, Q = quit

Publishes Twist to /cmd_vel at 10 Hz.  Run it in its own terminal - keystrokes
are read in cbreak mode (no Enter needed) and Ctrl+C works normally.

V = dig (lower the ladder, run the conveyor belt, fill the tub)
B = dump (open the gate, reverse the belt, build the berm)
N = stop the mechanism (and stow the ladder)
"""

import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

# Command magnitudes
LIN_SPEED = 0.30   # m/s
ANG_SPEED = 0.80   # rad/s

MSG = """
Keyboard teleop for the rover
-----------------------------
   W
A  S  D
   X

W = forward, S = backward, A = turn left, D = turn right
X / SPACE = stop driving, Q = quit

Excavation (the rover stops while the mechanism runs):
V = DIG    - lower the ladder, run the conveyor belt, fill the tub
B = DUMP   - open the gate, reverse the belt, build the berm
N = STOP   - stop the mechanism and stow the ladder

V and B are toggles: press the same key again to stop that work cycle.
Each key is a direct command: W/S drive straight, A/D turn in place.
The rover keeps its last drive command until you press another key.
"""


class TeleopKeyboard(Node):
    """Publishes Twist commands from WASD keyboard input."""

    def __init__(self) -> None:
        super().__init__("teleop_keyboard")
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.excavation_publisher = self.create_publisher(String, "/excavation/manual", 10)
        self.speed = 0.0
        self.turn = 0.0
        self.get_logger().info(MSG)

    def publish(self) -> None:
        msg = Twist()
        msg.linear.x = float(self.speed)
        msg.angular.z = float(self.turn)
        self.publisher.publish(msg)

    def send_excavation(self, command: str) -> None:
        """Stop the wheels, then ask module 6 for one manual work command."""
        self.speed = 0.0
        self.turn = 0.0
        self.publish()
        message = String()
        message.data = command
        self.excavation_publisher.publish(message)
        self.get_logger().info(f"excavation command: {command}")

    def handle_key(self, key: str) -> bool:
        """Update the command for one keystroke. Returns False to quit."""
        key = key.lower()
        if key == "w":
            self.speed = LIN_SPEED
            self.turn = 0.0
        elif key == "s":
            self.speed = -LIN_SPEED
            self.turn = 0.0
        elif key == "a":
            self.speed = 0.0
            self.turn = ANG_SPEED
        elif key == "d":
            self.speed = 0.0
            self.turn = -ANG_SPEED
        elif key in ("x", " "):
            self.speed = 0.0
            self.turn = 0.0
        elif key == "v":
            self.send_excavation("dig")
        elif key == "b":
            self.send_excavation("dump")
        elif key == "n":
            self.send_excavation("stop")
        elif key == "q":
            return False
        return True

    def run(self) -> None:
        if not sys.stdin.isatty():
            self.get_logger().error(
                "stdin is not a terminal. Run teleop from its own terminal:\n"
                "    ros2 run rover_description teleop_keyboard.py\n"
                "Do not start it from a launch file."
            )
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)  # no Enter needed, Ctrl+C still works
            running = True
            while rclpy.ok() and running:
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if readable:
                    key = sys.stdin.read(1)
                    running = self.handle_key(key)
                    self.get_logger().info(
                        f"speed={self.speed:+.2f} m/s  turn={self.turn:+.2f} rad/s",
                        throttle_duration_sec=0.2,
                    )
                self.publish()
        finally:
            self.speed = 0.0
            self.turn = 0.0
            self.publish()
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main() -> None:
    rclpy.init()
    node = TeleopKeyboard()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
