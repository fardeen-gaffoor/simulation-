#!/usr/bin/env python3
"""
Simple Teleop Node for Rover - with debug logging
Subscribes to /teleop_cmd (std_msgs/String) with commands: w, a, s, d, space
Publishes to /cmd_vel (geometry_msgs/Twist)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class SimpleTeleop(Node):
    def __init__(self):
        super().__init__('simple_teleop')
        
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            String, '/teleop_cmd', self.cmd_callback, 10)
        
        self.linear_speed = 1.0    # m/s
        self.angular_speed = 1.5   # rad/s
        
        # Key bindings
        self.key_bindings = {
            'w': (1.0, 0.0),   # forward
            's': (-1.0, 0.0),  # backward
            'a': (0.0, 1.5),   # turn left
            'd': (0.0, -1.5),  # turn right
            ' ': (0.0, 0.0),   # stop
        }
        
        self.cmd_count = 0
        self.get_logger().info("Simple Teleop started!")
        self.get_logger().info("Send commands via: ros2 topic pub /teleop_cmd std_msgs/String \"data: 'w'\"")
        self.get_logger().info("Commands: w=forward, s=backward, a=left, d=right, space=stop")

    def cmd_callback(self, msg: String):
        key = msg.data.strip().lower()
        self.cmd_count += 1
        
        self.get_logger().info(f"Received command #{self.cmd_count}: '{key}'")
        
        if key in self.key_bindings:
            linear, angular = self.key_bindings[key]
            twist = Twist()
            twist.linear.x = linear * 1.0
            twist.angular.z = angular * 1.5
            self.publisher.publish(twist)
            
            action = {
                'w': 'FORWARD',
                's': 'BACKWARD', 
                'a': 'LEFT',
                'd': 'RIGHT',
                ' ': 'STOP'
            }.get(key, key)
            
            self.get_logger().info(f"Command #{self.cmd_count}: {action} -> linear={twist.linear.x:.2f}, angular={twist.angular.z:.2f}")
        else:
            self.get_logger().warn(f"Unknown command: '{key}'")


def main():
    rclpy.init()
    node = SimpleTeleop()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Send stop command
        twist = Twist()
        node.publisher.publish(twist)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()