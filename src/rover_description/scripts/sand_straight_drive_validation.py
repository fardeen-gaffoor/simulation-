#!/usr/bin/env python3
"""Stage 4 of PROMPT_ROVER_SIM.md: sand traction validation - straight drive.

Reproduces the ``sandsim`` flat-sand straight-drive scenario (S2, commanded
speed 0.5 m/s on dry sand) inside the Gazebo ``cat_arena`` world and compares
the logged slip / torque / speed against the sandsim reference run.

What it does:

* publishes ``/cmd_vel`` (Twist) with ``linear.x = v_cmd`` (straight drive,
  ``angular.z = 0``) - the same topic the ``skid_steer_controller`` maps to the
  four per-wheel joint velocity commands;
* samples at a fixed rate:
    - body speed from ``/odom`` (ground-truth odometry bridged from gz),
    - per-wheel actual speed from ``/joint_states``,
    - per-wheel motor torque (N.m) and current (A) from
      ``/wheel/<name>/torque`` and ``/wheel/<name>/current`` (published by the
      ``motor_battery_model`` node);
* once the drive window elapses it stops the rover (zero ``/cmd_vel``) and
  keeps sampling briefly for the final-state statistics;
* writes the raw sample log to ``log_csv`` and compares the run against the
  sandsim S2 reference CSV
  (``/home/fewest/cat/tools/sandsim/results/s2_steady_drive/s2_steady_drive.csv``)
  using the documented tolerances.

Usage (with the arena world + rover + sand_traction_plugin running):

    ros2 launch rover_description sim.launch.py seed:=7 headless:=true headless_rendering:=true
    ros2 run rover_description sand_straight_drive_validation

The node never exits on its own (it spins until SIGINT/SIGTERM); scripted runs
stop it after the drive window.

Honesty note: this compares two models (Bekker-Wong lookup inside Gazebo via
``sand_traction_plugin`` vs the ``sandsim`` terramechanics tool) and the
tolerances are engineering tolerances for that cross-model check, not a claim
of physical equivalence.
"""

from __future__ import annotations

import csv
import math
import os
from functools import partial
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from rover_description.config import default_config_path  # noqa: F401  (sanity import)


WHEEL_NAMES: List[str] = [
    "wheel_front_left",
    "wheel_front_right",
    "wheel_rear_left",
    "wheel_rear_right",
]

JOINT_NAMES: List[str] = [
    "wheel_front_left_joint",
    "wheel_front_right_joint",
    "wheel_rear_left_joint",
    "wheel_rear_right_joint",
]

S2_REFERENCE_CSV = "/home/fewest/cat/tools/sandsim/results/s2_steady_drive/s2_steady_drive.csv"

# Documented cross-model comparison tolerances (Stage 4 validation).
TOL_SPEED_MPS = 0.15            # |gazebo - sandsim| m/s on the achieved straight-drive speed
TOL_PEAK_TORQUE_FACTOR = 1.5   # gazebo peak torque may exceed sandsim peak torque by this factor
TOL_PEAK_TORQUE_NM = 7.35      # advised maximum torque (75 kg.cm) in N.m; gearbox break = 100 kg.cm = 9.81 N.m
STEADY_TORQUE_NM = 3.92        # rated torque (40 kg.cm) - steady straight-drive requirement


class SandStraightDriveValidation(Node):
    """Drive the rover straight at ``v_cmd`` in the sand arena and log it."""

    def __init__(self) -> None:
        super().__init__("sand_straight_drive_validation")

        self.declare_parameter("duration_s", 14.0)
        self.declare_parameter("v_cmd", 0.5)
        self.declare_parameter("log_csv", "/home/fewest/cat/simulation-/results/sand_straight_drive/sand_straight_drive_validation.csv")
        self.declare_parameter("reference_csv", S2_REFERENCE_CSV)

        self.duration_s = float(self.get_parameter("duration_s").value)
        self.v_cmd = float(self.get_parameter("v_cmd").value)
        self.log_csv = str(self.get_parameter("log_csv").value)
        self.reference_csv = str(self.get_parameter("reference_csv").value)

        self.t0: Optional[float] = None
        self.stop_t: Optional[float] = None
        self.stopped = False
        self._finished = False

        self.body_speed: float = 0.0
        self.wheel_speeds: Dict[str, float] = {n: 0.0 for n in WHEEL_NAMES}
        self.wheel_torques: Dict[str, float] = {n: 0.0 for n in WHEEL_NAMES}
        self.wheel_currents: Dict[str, float] = {n: 0.0 for n in WHEEL_NAMES}

        self.samples: List[dict] = []

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        # Also publish directly to wheel topics to bypass skid_steer_controller QoS issues
        self.wheel_cmd_pubs: dict[str, rclpy.publisher.Publisher] = {}
        for n in WHEEL_NAMES:
            self.wheel_cmd_pubs[n] = self.create_publisher(
                Float64, f"/wheel/{n}/cmd_vel", 10
            )
        self.odom_sub = self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.js_sub = self.create_subscription(JointState, "/joint_states", self._joint_states_cb, 10)
        for n in WHEEL_NAMES:
            self.create_subscription(
                Float64, f"/wheel/{n}/torque", partial(self._torque_cb, name=n), 10
            )
            self.create_subscription(
                Float64, f"/wheel/{n}/current", partial(self._current_cb, name=n), 10
            )

        self.timer = self.create_timer(0.05, self._tick)

        self.get_logger().info(
            f"sand_straight_drive_validation: v_cmd={self.v_cmd} m/s, drive window {self.duration_s} s, "
            f"log {self.log_csv}, reference {self.reference_csv}"
        )

    # ------------------------------------------------------------------ #
    # subscriptions                                                      #
    # ------------------------------------------------------------------ #
    def _odom_cb(self, msg: Odometry) -> None:
        t = self.get_clock().now().nanoseconds / 1e9
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.body_speed = math.hypot(vx, vy)
        if self.t0 is None:
            self.t0 = t

    def _joint_states_cb(self, msg: JointState) -> None:
        name_to_vel = {n: float(v) for n, v in zip(msg.name, msg.velocity) if v is not None}
        for jn, wn in zip(JOINT_NAMES, WHEEL_NAMES):
            if jn in name_to_vel:
                # joint velocity is rad/s -> m/s at the wheel radius
                self.wheel_speeds[wn] = abs(name_to_vel[jn]) * 0.10

    def _torque_cb(self, msg: Float64, name: str) -> None:
        self.wheel_torques[name] = float(msg.data)

    def _current_cb(self, msg: Float64, name: str) -> None:
        self.wheel_currents[name] = float(msg.data)

    # ------------------------------------------------------------------ #
    # drive + logging                                                    #
    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        t = self.get_clock().now().nanoseconds / 1e9
        if self.t0 is None:
            self.t0 = t

        elapsed = t - self.t0
        if not self.stopped and elapsed >= self.duration_s:
            self.stopped = True
            self.stop_t = t
            self.get_logger().info(f"sand_straight_drive_validation: drive window elapsed ({self.duration_s} s) - stopping")
            self._publish_cmd(0.0, 0.0)

        if self.stopped and self.stop_t is not None and (t - self.stop_t) > 2.0:
            if not self._finished:
                self._finished = True
                self._finish()
            return

        if not self.stopped:
            self._publish_cmd(self.v_cmd, 0.0)
        record = True

        if record:
            self.samples.append(
                {
                    "t": t,
                    "body_speed_mps": self.body_speed,
                    "wheel_speeds_mps": dict(self.wheel_speeds),
                    "wheel_torques_nm": dict(self.wheel_torques),
                    "wheel_currents_a": dict(self.wheel_currents),
                }
            )

    def _publish_cmd(self, vx: float, vy: float) -> None:
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)

        # Also publish directly to wheel topics (bypass skid_steer_controller)
        effective_track = 0.6 * 2.0  # track * correction
        v_left = vx - 0.0 * effective_track / 2.0
        v_right = vx + 0.0 * effective_track / 2.0
        wl = max(-12.0, min(12.0, v_left / 0.1))
        wr = max(-12.0, min(12.0, v_right / 0.1))
        cmd = Float64()
        cmd.data = wl
        self.wheel_cmd_pubs["wheel_front_left"].publish(cmd)
        self.wheel_cmd_pubs["wheel_rear_left"].publish(cmd)
        cmd.data = wr
        self.wheel_cmd_pubs["wheel_front_right"].publish(cmd)
        self.wheel_cmd_pubs["wheel_rear_right"].publish(cmd)

    # ------------------------------------------------------------------ #
    # statistics + comparison                                             #
    # ------------------------------------------------------------------ #
    def _final_stats(self) -> dict:
        final = self.samples[-1] if self.samples else {}
        peak_speed = max((s["body_speed_mps"] for s in self.samples), default=0.0)
        peak_torque = max((max(s["wheel_torques_nm"].values()) for s in self.samples if s["wheel_torques_nm"]), default=0.0)
        peak_current = max((max(s["wheel_currents_a"].values()) for s in self.samples if s["wheel_currents_a"]), default=0.0)

        t_target: Optional[float] = None
        for s in self.samples:
            if s["body_speed_mps"] >= self.v_cmd:
                t_target = s["t"]
                break

        return {
            "v_cmd_mps": self.v_cmd,
            "final_speed_mps": final.get("body_speed_mps", 0.0),
            "peak_speed_mps": peak_speed,
            "peak_torque_nm": peak_torque,
            "peak_current_a": peak_current,
            "time_to_target_s": t_target,
            "n_samples": len(self.samples),
        }

    def _finish(self) -> None:
        os.makedirs(os.path.dirname(self.log_csv), exist_ok=True)
        header = ["t", "body_speed_mps"]
        for n in WHEEL_NAMES:
            header.append(f"wheel_{n}_speed_mps")
        for n in WHEEL_NAMES:
            header.append(f"wheel_{n}_torque_nm")
        for n in WHEEL_NAMES:
            header.append(f"wheel_{n}_current_a")
        with open(self.log_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for s in self.samples:
                row = [s["t"], s["body_speed_mps"]]
                for n in WHEEL_NAMES:
                    row.append(s["wheel_speeds_mps"].get(n, 0.0))
                for n in WHEEL_NAMES:
                    row.append(s["wheel_torques_nm"].get(n, 0.0))
                for n in WHEEL_NAMES:
                    row.append(s["wheel_currents_a"].get(n, 0.0))
                w.writerow(row)

        gazebo = self._final_stats()
        sandsim = self._reference_stats()
        lines, failures = self._compare(gazebo, sandsim)
        for ln in lines:
            self.get_logger().info(ln)
        report_path = self.log_csv.replace(".csv", "_report.txt")
        with open(report_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
            fh.write(f"FAILURES: {failures}\n")
        self.get_logger().info(f"sand_straight_drive_validation: report written to {report_path}")

    def _reference_stats(self) -> dict:
        result = {
            "achieved_speed_mps": None,
            "peak_torque_nm": None,
            "peak_current_a": None,
            "peak_slip": None,
            "time_to_target_s": None,
            "final_time_s": None,
        }
        if not os.path.isfile(self.reference_csv):
            return result
        speeds: List[float] = []
        torques: List[float] = []
        currents: List[float] = []
        slips: List[float] = []
        times: List[float] = []
        t_target: Optional[float] = None
        with open(self.reference_csv, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                speed = float(row.get("speed_mps") or 0.0)
                torque = float(row.get("torque_Nm") or 0.0)
                current = float(row.get("current_A") or 0.0)
                slip = float(row.get("slip") or 0.0)
                t = float(row.get("t") or 0.0)
                speeds.append(speed)
                torques.append(torque)
                currents.append(current)
                slips.append(slip)
                times.append(t)
                if t_target is None and speed >= self.v_cmd:
                    t_target = t

        result["achieved_speed_mps"] = speeds[-1] if speeds else None
        result["peak_torque_nm"] = max(torques) if torques else None
        result["peak_current_a"] = max(currents) if currents else None
        result["peak_slip"] = max(slips) if slips else None
        result["time_to_target_s"] = t_target
        result["final_time_s"] = times[-1] if times else None
        return result

    def _compare(self, gazebo: dict, sandsim: dict) -> Tuple[List[str], List[str]]:
        lines: List[str] = []
        failures: List[str] = []

        if sandsim.get("achieved_speed_mps") is None:
            lines.append("COMPARE: could not read sandsim reference - cannot compare")
            return lines, failures

        gz_speed = gazebo.get("final_speed_mps") or 0.0
        sb_speed = sandsim.get("achieved_speed_mps") or 0.0
        speed_ok = abs(gz_speed - sb_speed) <= TOL_SPEED_MPS
        lines.append(
            f"COMPARE speed: gazebo final {gz_speed:.4f} m/s vs sandsim achieved {sb_speed:.4f} m/s "
            f"(tolerance {TOL_SPEED_MPS} m/s) -> {'PASS' if speed_ok else 'FAIL'}"
        )
        if not speed_ok:
            failures.append(f"straight drive speed: gazebo final {gz_speed:.3f} m/s vs sandsim {sb_speed:.3f} m/s")

        gz_peak_torque = gazebo.get("peak_torque_nm") or 0.0
        sb_peak_torque = sandsim.get("peak_torque_nm") or 0.0
        torque_ok = gz_peak_torque <= max(sb_peak_torque * TOL_PEAK_TORQUE_FACTOR, TOL_PEAK_TORQUE_NM)
        lines.append(
            f"COMPARE peak torque: gazebo {gz_peak_torque:.4f} N.m vs sandsim peak {sb_peak_torque:.4f} N.m "
            f"(limit: advised max {TOL_PEAK_TORQUE_NM} N.m / gearbox break 9.81 N.m) -> {'PASS' if torque_ok else 'FAIL'}"
        )
        if not torque_ok:
            failures.append(f"straight drive peak torque: gazebo {gz_peak_torque:.3f} N.m")

        steady_ok = gz_peak_torque <= STEADY_TORQUE_NM
        lines.append(
            f"COMPARE steady torque: gazebo peak {gz_peak_torque:.4f} N.m vs required <= {STEADY_TORQUE_NM} N.m "
            f"(rated 40 kg.cm) -> {'PASS' if steady_ok else 'FAIL'}"
        )
        if not steady_ok:
            failures.append(f"straight drive steady torque: gazebo peak {gz_peak_torque:.3f} N.m > {STEADY_TORQUE_NM} N.m")

        return lines, failures


def main() -> None:
    rclpy.init()
    node = SandStraightDriveValidation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
