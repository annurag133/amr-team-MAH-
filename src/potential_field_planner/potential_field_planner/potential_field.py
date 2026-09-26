import math
import signal

import numpy as np
import rclpy
import rclpy.time
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)

import tf2_ros
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty, String

from potential_field_planner.scan_utils import planar_transform, project_scan


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class PotentialFieldPlanner(Node):

    def __init__(self):
        super().__init__('potential_field_planner')

        params = {
            'base_frame': 'base_link',
            'global_frame': 'map',
            'rate': 20.0,
            'waypoint_tolerance': 0.30,
            'goal_tolerance': 0.10,
            'yaw_tolerance': 0.10,
            'align_final_yaw': True,
            'k_att': 1.0,
            'k_rep': 0.5,
            'max_rep_ratio': 0.8,
            'k_tan': 1.0,
            'holonomic_final_approach': True,
            'collision_margin': 0.04,
            'final_approach_dist': 0.5,
            'final_approach_speed': 0.15,
            'influence_dist': 0.8,
            'max_lin': 0.25,
            'max_ang': 0.5,
            'k_ang': 1.5,
            'max_lin_acc': 0.5,
            'max_ang_acc': 2.0,
            'robot_half_length': 0.35,
            'robot_half_width': 0.24,
            'stop_dist': 0.15,
            'slow_dist': 0.50,
            'scan_timeout': 0.5,
            'tf_timeout': 0.5,
            'stuck_timeout': 8.0,
            'stuck_progress': 0.15,
            'max_replans': 3,
        }
        for k, v in params.items():
            self.declare_parameter(k, v)

        self.scan = None
        self.scan_pts = None
        self.path = []
        self.goal_yaw = None
        self.idx = 0
        self.state = 'IDLE'
        self.last_cmd = Twist()
        self.last_time = None
        self.best_dist = math.inf
        self.progress_time = None
        self.replans = 0
        self.replan_pending = False
        self.laser_tf = None
        self.tan_sign = 0
        self.front_rep = (0.0, 0.0)
        self.min_obstacle_dist = math.inf
        self.last_tf_stamp = None
        self.last_tf_change = None

        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(LaserScan, '/scan', self.scan_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Path, '/global_path', self.path_callback, latched)
        self.create_subscription(Empty, '/cancel_goal', self.cancel_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.replan_pub = self.create_publisher(Empty, '/replan', 10)
        self.status_pub = self.create_publisher(String, '/navigation_status', latched)

        self.timer = self.create_timer(1.0 / self.p('rate'), self.control_loop)
        self.set_state('IDLE')
        self.get_logger().info('Potential Field Planner started')

    def p(self, name):
        return self.get_parameter(name).value

    def set_state(self, state, reason=''):
        if state != self.state or reason:
            text = f'State {self.state} -> {state}' + (f': {reason}' if reason else '')
            if state in ('BLOCKED', 'FAILED', 'WAITING'):
                self.get_logger().warn(text)
            else:
                self.get_logger().info(text)
        self.state = state
        self.status_pub.publish(String(data=state))

    # ------------------------------------------------------------ inputs

    def scan_callback(self, msg):
        if self.laser_tf is None or self.laser_tf[0] != msg.header.frame_id:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.p('base_frame'), msg.header.frame_id, rclpy.time.Time())
                self.laser_tf = (msg.header.frame_id, planar_transform(t))
            except Exception:
                return

        r = np.asarray(msg.ranges, dtype=float)
        a = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        valid = np.isfinite(r) & (r > max(msg.range_min, 0.02)) & (r < msg.range_max)
        px, py = project_scan(r[valid], a[valid], self.laser_tf[1])
        # Drop returns from the robot's own body
        body = (np.abs(px) < self.p('robot_half_length') * 0.9) & \
               (np.abs(py) < self.p('robot_half_width') * 0.9)
        self.scan_pts = (px[~body], py[~body])
        self.scan = msg
        self.scan_time = self.get_clock().now()

    def path_callback(self, msg):
        if not msg.poses:
            if self.state not in ('IDLE', 'FAILED'):
                self.set_state('FAILED', 'planner returned no path')
            self.path = []
            self.stop()
            return
        self.path = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]
        q = msg.poses[-1].pose.orientation
        self.goal_yaw = yaw_of(q)
        self.idx = 1 if len(self.path) > 1 else 0
        self.progress_time = self.get_clock().now()
        if not self.replan_pending:
            self.replans = 0
            self.best_dist = math.inf
        self.replan_pending = False
        self.tan_sign = 0
        self.set_state('FOLLOWING', f'{len(self.path)} waypoints')

    def cancel_callback(self, _):
        self.path = []
        self.replans = 0
        self.stop()
        self.set_state('IDLE', 'cancelled')

    def robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.p('global_frame'), self.p('base_frame'), rclpy.time.Time())
        except Exception:
            return None
        # Staleness = stamp stopped advancing (robust to robot/laptop clock offset)
        stamp = (t.header.stamp.sec, t.header.stamp.nanosec)
        now = self.get_clock().now()
        if stamp != self.last_tf_stamp:
            self.last_tf_stamp = stamp
            self.last_tf_change = now
        elif stamp[0] != 0 and (now - self.last_tf_change) > \
                Duration(seconds=self.p('tf_timeout') + 1.0):
            return None
        return (t.transform.translation.x, t.transform.translation.y,
                yaw_of(t.transform.rotation))

    # ------------------------------------------------------------ control

    def stop(self):
        self.last_cmd = Twist()
        self.cmd_pub.publish(Twist())

    def publish_limited(self, v, w, vy=0.0):
        now = self.get_clock().now()
        dt = 1.0 / self.p('rate')
        if self.last_time is not None:
            dt = min(max((now - self.last_time).nanoseconds / 1e9, 1e-3), 0.2)
        self.last_time = now
        dv = self.p('max_lin_acc') * dt
        dw = self.p('max_ang_acc') * dt
        # Only acceleration is limited; braking is immediate
        v = float(min(v, self.last_cmd.linear.x + dv))
        w = float(np.clip(w, self.last_cmd.angular.z - dw, self.last_cmd.angular.z + dw))
        vy = float(np.clip(vy, self.last_cmd.linear.y - dv, self.last_cmd.linear.y + dv))
        v, vy, w = self.collision_filter(v, vy, w)
        cmd = Twist()
        cmd.linear.x = v
        cmd.linear.y = vy
        cmd.angular.z = w
        self.last_cmd = cmd
        self.cmd_pub.publish(cmd)

    def footprint_dist(self, x, y):
        hl, hw = self.p('robot_half_length'), self.p('robot_half_width')
        return np.hypot(np.maximum(np.abs(x) - hl, 0.0), np.maximum(np.abs(y) - hw, 0.0))

    def would_collide(self, vx, vy, w):
        # Simulate the next half second of this command against the current laser
        # points; a collision is a point getting closer than the margin to the
        # robot's outline (points we are already moving away from are fine).
        if self.scan_pts is None:
            return False
        px, py = self.scan_pts
        near = np.hypot(px, py) < 1.5
        px, py = px[near], py[near]
        if px.size == 0:
            return False
        d_now = self.footprint_dist(px, py)
        margin = self.p('collision_margin')
        for t in (0.15, 0.3, 0.5):
            th = w * t
            tx = vx * t * math.cos(th / 2) - vy * t * math.sin(th / 2)
            ty = vx * t * math.sin(th / 2) + vy * t * math.cos(th / 2)
            c, s = math.cos(-th), math.sin(-th)
            qx = c * (px - tx) - s * (py - ty)
            qy = s * (px - tx) + c * (py - ty)
            d = self.footprint_dist(qx, qy)
            if np.any((d < margin) & (d < d_now - 1e-3)):
                return True
        return False

    def collision_filter(self, v, vy, w):
        if not self.would_collide(v, vy, w):
            return v, vy, w
        for cand in ((v, vy, 0.0), (0.0, 0.0, w)):
            if (cand[0] or cand[1] or cand[2]) and not self.would_collide(*cand):
                self.get_logger().warn('Collision predicted, dropping part of the command',
                                       throttle_duration_sec=2.0)
                return cand
        self.get_logger().warn('Collision predicted, stopping', throttle_duration_sec=2.0)
        return 0.0, 0.0, 0.0

    def final_approach(self, dx, dy, yaw, dist):
        # Last stretch: slide straight onto the goal (Robile is omnidirectional)
        # instead of steering like a car, which dithers left/right near the goal.
        # A* already guarantees this stretch is free of mapped obstacles.
        c, s = math.cos(yaw), math.sin(yaw)
        ex, ey = c * dx + s * dy, -s * dx + c * dy
        speed = min(self.p('final_approach_speed'), 0.8 * dist + 0.03)
        vx, vy = speed * ex / max(dist, 1e-6), speed * ey / max(dist, 1e-6)
        # Never slide into something: check clearance in the direction of motion
        if self.clearance_towards(vx, vy) < self.p('stop_dist'):
            vx = vy = 0.0
        w = 0.0
        # Turning sweeps the corners, so only rotate with some room all around
        if self.p('align_final_yaw') and self.goal_yaw is not None and \
                self.min_obstacle_dist > 0.08:
            w = float(np.clip(self.p('k_ang') * wrap(self.goal_yaw - yaw), -0.4, 0.4))
        self.publish_limited(vx, w, vy)

    def clearance_towards(self, vx, vy):
        # Footprint distance to laser points lying in the direction of travel
        if abs(vx) + abs(vy) < 1e-6:
            return math.inf
        px, py = self.scan_pts
        hl, hw = self.p('robot_half_length'), self.p('robot_half_width')
        ahead = (px * vx + py * vy) > 0
        if not np.any(ahead):
            return math.inf
        ex = np.maximum(np.abs(px[ahead]) - hl, 0.0)
        ey = np.maximum(np.abs(py[ahead]) - hw, 0.0)
        return float(np.hypot(ex, ey).min())

    def repulsion(self):
        px, py = self.scan_pts
        hl, hw = self.p('robot_half_length'), self.p('robot_half_width')
        # Distance from the robot's rectangular footprint, not its centre
        ex = np.maximum(np.abs(px) - hl, 0.0)
        ey = np.maximum(np.abs(py) - hw, 0.0)
        d = np.hypot(ex, ey)
        self.min_obstacle_dist = float(d.min()) if d.size else math.inf
        d0 = self.p('influence_dist')
        near = d < d0
        self.front_rep = (0.0, 0.0)
        if not np.any(near):
            return 0.0, 0.0
        d = np.maximum(d[near], 0.02)
        nx, ny = px[near], py[near]
        norm = np.maximum(np.hypot(nx, ny), 1e-3)
        # Integrate over the scan so the result doesn't depend on beam count
        w = self.p('k_rep') * (1.0 / d - 1.0 / d0) / d * self.scan.angle_increment
        rx = float(-np.sum(w * nx / norm))
        ry = float(-np.sum(w * ny / norm))
        front = (nx > 0) & (np.abs(np.arctan2(ny, nx)) < 1.0)
        fr = (float(-np.sum(w[front] * nx[front] / norm[front])),
              float(-np.sum(w[front] * ny[front] / norm[front])))
        fm = math.hypot(*fr)
        limit_f = self.p('max_rep_ratio') * self.p('k_att')
        self.front_rep = (fr[0] * limit_f / fm, fr[1] * limit_f / fm) if fm > limit_f else fr
        # Bounded below the attraction, so obstacles steer the robot but can
        # never cancel the goal (no local minima); the clearance stop handles safety.
        limit = self.p('max_rep_ratio') * self.p('k_att')
        mag = math.hypot(rx, ry)
        if mag > limit:
            rx, ry = rx * limit / mag, ry * limit / mag
        return rx, ry

    def tangential(self, att_x, att_y):
        # Rotate the repulsion of obstacles *ahead* by 90 deg so the robot slides
        # around them instead of stalling (potential field local minima). Walls
        # beside the robot don't contribute.
        rep_x, rep_y = self.front_rep
        mag = math.hypot(rep_x, rep_y)
        if mag < 1e-6:
            self.tan_sign = 0
            return 0.0, 0.0
        tx, ty = -rep_y, rep_x
        dot = (tx * att_x + ty * att_y) / (mag * max(math.hypot(att_x, att_y), 1e-6))
        # Hysteresis: keep the chosen side unless it clearly leads away from the goal
        if self.tan_sign == 0 or self.tan_sign * dot < -0.3:
            self.tan_sign = 1 if dot >= 0 else -1
        k = self.p('k_tan')
        return k * self.tan_sign * tx, k * self.tan_sign * ty

    def front_clearance(self, direction):
        # Free distance along the driving direction inside the robot's width
        px, py = self.scan_pts
        hl, hw = self.p('robot_half_length'), self.p('robot_half_width')
        sel = (np.sign(direction) * px > hl) & (np.abs(py) < hw + 0.05)
        if not np.any(sel):
            return math.inf
        return float(np.min(np.abs(px[sel])) - hl)

    def check_progress(self, dist_to_goal):
        now = self.get_clock().now()
        if dist_to_goal < self.best_dist - self.p('stuck_progress'):
            self.best_dist = dist_to_goal
            self.progress_time = now
            self.replans = 0
            return
        if (now - self.progress_time) > Duration(seconds=self.p('stuck_timeout')):
            self.stop()
            if self.replans >= self.p('max_replans'):
                self.path = []
                self.set_state('FAILED', 'no progress after replanning, giving up')
                return
            self.replans += 1
            self.set_state('BLOCKED', f'no progress, replan {self.replans}/{self.p("max_replans")}')
            self.progress_time = now
            self.replan_pending = True
            self.replan_pub.publish(Empty())

    def control_loop(self):
        if not self.path or self.state in ('IDLE', 'FAILED', 'REACHED'):
            return

        if self.scan is None or (self.get_clock().now() - self.scan_time) > \
                Duration(seconds=self.p('scan_timeout')):
            self.stop()
            if self.state != 'WAITING':
                self.set_state('WAITING', 'laser scan missing or stale')
            return

        pose = self.robot_pose()
        if pose is None:
            self.stop()
            if self.state != 'WAITING':
                self.set_state('WAITING', 'no localisation (map->base TF)')
            return
        if self.state == 'WAITING':
            self.set_state('FOLLOWING', 'inputs recovered')
            self.progress_time = self.get_clock().now()

        x, y, yaw = pose
        last = len(self.path) - 1
        gx, gy = self.path[last]
        dist_goal = math.hypot(gx - x, gy - y)

        if self.state == 'ALIGNING' or dist_goal < self.p('goal_tolerance'):
            if self.p('align_final_yaw') and self.goal_yaw is not None:
                err = wrap(self.goal_yaw - yaw)
                if abs(err) > self.p('yaw_tolerance'):
                    if self.state != 'ALIGNING':
                        self.set_state('ALIGNING')
                    w = float(np.clip(self.p('k_ang') * err, -self.p('max_ang'), self.p('max_ang')))
                    if abs(w) < 0.15:
                        w = math.copysign(0.15, w)
                    self.publish_limited(0.0, w)
                    return
            self.stop()
            self.path = []
            self.set_state('REACHED', f'goal reached ({dist_goal:.2f} m off)')
            return

        # Advance past reached waypoints, including ones we overshot
        while self.idx < last:
            wx, wy = self.path[self.idx]
            nx, ny = self.path[self.idx + 1]
            d_cur = math.hypot(wx - x, wy - y)
            if d_cur < self.p('waypoint_tolerance') or math.hypot(nx - x, ny - y) < d_cur:
                self.idx += 1
            else:
                break

        remaining = math.hypot(self.path[self.idx][0] - x, self.path[self.idx][1] - y) + \
            sum(math.hypot(self.path[i + 1][0] - self.path[i][0], self.path[i + 1][1] - self.path[i][1])
                for i in range(self.idx, last))
        self.check_progress(remaining)
        if self.state in ('FAILED',):
            return
        if self.state == 'BLOCKED':
            self.set_state('FOLLOWING')

        if self.p('holonomic_final_approach') and self.idx == last and \
                dist_goal < self.p('final_approach_dist'):
            self.final_approach(gx - x, gy - y, yaw, dist_goal)
            return

        tx, ty = self.path[self.idx]
        dx, dy = tx - x, ty - y
        d = max(math.hypot(dx, dy), 1e-6)
        c, s = math.cos(yaw), math.sin(yaw)
        att_x = self.p('k_att') * (c * dx + s * dy) / d
        att_y = self.p('k_att') * (-s * dx + c * dy) / d

        rep_x, rep_y = self.repulsion()
        tan_x, tan_y = self.tangential(att_x, att_y)
        fx, fy = att_x + rep_x + tan_x, att_y + rep_y + tan_y
        angle = math.atan2(fy, fx)

        clearance = self.front_clearance(1.0)
        speed = self.p('max_lin') * max(0.0, math.cos(angle)) ** 2
        speed *= float(np.clip((clearance - self.p('stop_dist')) /
                               max(self.p('slow_dist') - self.p('stop_dist'), 1e-3), 0.0, 1.0))
        # Also slow down when anything is close on any side (corners sweep when turning)
        speed *= float(np.clip((self.min_obstacle_dist - 0.05) / 0.35, 0.3, 1.0))
        if self.idx == last:
            speed *= min(1.0, 0.2 + dist_goal / 0.5)
        if clearance < self.p('stop_dist'):
            # Blocked ahead: turn towards the waypoint rather than dithering on the field
            angle = math.atan2(att_y, att_x)
        # Target (nearly) behind: keep turning the same way instead of flipping
        # between +180 and -180 degrees every cycle
        if abs(angle) > 2.6 and self.last_cmd.angular.z != 0.0 and \
                math.copysign(1.0, angle) != math.copysign(1.0, self.last_cmd.angular.z):
            angle = math.copysign(abs(angle), self.last_cmd.angular.z)
        w = float(np.clip(self.p('k_ang') * angle, -self.p('max_ang'), self.p('max_ang')))

        self.publish_limited(speed, w)


def main(args=None):
    rclpy.init(args=args)
    node = PotentialFieldPlanner()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        # Context torn down by Ctrl+C mid-spin is a normal shutdown, not an error
        if rclpy.ok():
            raise
    finally:
        # Already shutting down: ignore the second Ctrl+C (terminal + launch both send one)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # Never leave the robot driving on exit
        try:
            node.cmd_pub.publish(Twist())
        except Exception:
            pass
        # A second Ctrl+C (terminal + launch both send one) must not print a traceback
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, ExternalShutdownException, Exception):
            pass


if __name__ == '__main__':
    main()
