#!/usr/bin/env python3
"""
WASD Teleoperation for Rover
W - Forward
S - Backward
A - Turn Left
D - Turn Right
Space - Stop
Q - Quit
"""

import sys
import tty
import termios
import select
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import threading


class WASDTeleop(Node):
    def __init__(self):
        super().__init__('wasd_teleop')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.linear_speed = 1.0    # m/s
        self.angular_speed = 1.5   # rad/s
        self.running = True
        
        # Key bindings
        self.key_bindings = {
            'w': (1.0, 0.0),   # forward
            's': (-1.0, 0.0),  # backward
            'a': (0.0, 1.5),   # turn left
            'd': (0.0, -1.5),  # turn right
            ' ': (0.0, 0.0),   # stop
        }
        
        self.get_logger().info("WASD Teleop started!")
        self.get_logger().info("Controls:")
        self.get_logger().info("  W - Forward")
        self.get_logger().info("  S - Backward")
        self.get_logger().info("  A - Turn Left")
        self.get_logger().info("  D - Turn Right")
        self.get_logger().info("  Space - Stop")
        self.get_logger().info("  Q - Quit")
        self.get_logger().info("Press keys to control the rover...")

    def get_key(self):
        """Get a single keypress without blocking."""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def run(self):
        """Main teleop loop."""
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            twist = Twist()
            
            while self.running and rclpy.ok():
                key = self.get_key()
                
                if key == 'q' or key == 'Q':
                    self.get_logger().info("Quitting...")
                    self.running = False
                    break
                    
                if key in self.key_bindings:
                    linear, angular = self.key_bindings[key]
                    twist.linear.x = linear * self.linear_speed
                    twist.angular.z = angular * self.angular_speed
                    self.publisher.publish(twist)
                    
                    if key == ' ':
                        self.get_logger().info("STOP")
                    else:
                        self.get_logger().debug(f"Key: {key} -> linear={twist.linear.x:.2f}, angular={twist.angular.z:.2f}")
                
                rclpy.spin_once(self, timeout_sec=0.05)
                
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            # Send stop command
            twist = Twist()
            self.publisher.publish(twist)
            self.get_logger().info("Teleop stopped")


def main():
    rclpy.init()
    
    # Save terminal settings
    global old_settings
    old_settings = termios.tcgetattr(sys.stdin)
    
    teleop = WASDTeleop()
    
    try:
        teleop.run()
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        rclpy.shutdown()


if __name__ == '__main__':
    main()