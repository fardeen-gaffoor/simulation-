#!/usr/bin/env python3
"""
Test script for WASD teleop - runs everything in one process
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import time
import threading


class TestTeleop(Node):
    def __init__(self):
        super().__init__('test_teleop')
        
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            String, '/teleop_cmd', self.cmd_callback, 10)
        
        self.linear_speed = 1.0
        self.angular_speed = 1.5
        self.key_bindings = {
            'w': (1.0, 0.0), 's': (-1.0, 0.0),
            'a': (0.0, 1.5), 'd': (0.0, -1.5),
            ' ': (0.0, 0.0)
        }
        self.received_commands = []
        
        self.get_logger().info("Test Teleop started!")
        
    def cmd_callback(self, msg: String):
        key = msg.data.strip().lower()
        self.get_logger().info(f"Received: '{key}'")
        
        if key in self.key_bindings:
            linear, angular = self.key_bindings[key]
            twist = Twist()
            twist.linear.x = linear * 1.0
            twist.angular.z = angular * 1.5
            self.pub.publish(twist)
            self.get_logger().info(f"Published: linear={twist.linear.x:.2f}, angular={twist.angular.z:.2f}")
        else:
            self.get_logger().warn(f"Unknown key: '{key}'")


def test_teleop():
    rclpy.init()
    
    node = Node('test_publisher')
    pub = node.create_publisher(String, '/teleop_cmd', 10)
    
    # Wait for publisher to be ready
    time.sleep(1.0)
    
    # Send test commands
    commands = ['w', 'a', 's', 'd', ' ']
    
    for cmd in commands:
        msg = String()
        msg.data = cmd
        node.get_logger().info(f"Publishing command: '{cmd}'")
        pub.publish(msg)
        time.sleep(0.5)
    
    time.sleep(1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    import time
    from std_msgs.msg import String
    
    rclpy.init()
    
    # Create teleop node in background thread
    teleop_node = TestTeleop()
    
    # Spin teleop node in background
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(teleop_node)
    
    def spin_executor():
        executor.spin()
    
    import threading
    spin_thread = threading.Thread(target=spin_executor, daemon=True)
    spin_thread.start()
    
    # Wait for teleop to be ready
    time.sleep(1.0)
    
    # Create publisher node
    pub_node = Node('test_publisher')
    pub = pub_node.create_publisher(String, '/teleop_cmd', 10)
    time.sleep(0.5)
    
    # Send test commands
    commands = ['w', 'a', 's', 'd', ' ']
    for cmd in commands:
        msg = String()
        msg.data = cmd
        print(f"Sending: '{cmd}'")
        pub.publish(msg)
        time.sleep(0.5)
    
    time.sleep(1.0)
    
    rclpy.shutdown()


if __name__ == '__main__':
    test_teleop()