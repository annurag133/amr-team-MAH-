"""Frontier-based exploration for the Robile.

Works on the live SLAM map: finds frontiers (known-free cells bordering unknown
space), picks the best reachable one and sends it as /goal_pose to the A* +
potential field navigation stack. Repeats until no frontiers are left, then
saves the map.
"""
import math
import signal
import os
import subprocess

import numpy as np
import rclpy
import rclpy.time
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.ndimage import binary_dilation, distance_transform_edt, label

import tf2_ros
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Empty, String
from visualization_msgs.msg import Marker, MarkerArray

EIGHT = np.ones((3, 3), dtype=bool)
FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def geodesic_distance(passable, start, max_iter=5000):
    """Travel distance (cells, 4-connected) from start over passable cells; inf if unreachable."""
    dist = np.full(passable.shape, np.inf)
    front = np.zeros_like(passable)
    front[start[1], start[0]] = True
    seen = front.copy()
    dist[front] = 0
    for i in range(1, max_iter):
        grown = binary_dilation(front, FOUR) & passable & ~seen
        if not grown.any():
            break
        dist[grown] = i
        seen |= grown
        front = grown
    return dist


class FrontierExplorer(Node):

    def __init__(self):
        super().__init__('frontier_explorer')
        params = {
            'global_frame': 'map',
            'base_frame': 'base_link',
            'robot_radius': 0.30,         # planning radius, must match astar_global_planner
            'goal_clearance': 0.45,       # goals need room to turn in place (half-diagonal 0.42)
            'min_frontier_size': 0.4,     # m of frontier length to be worth visiting
            'goal_search_radius': 1.0,    # how far from a frontier the safe goal may be
            'info_gain_weight': 2.0,      # value of frontier length vs travel distance
            'min_goal_dist': 0.5,         # skip frontiers closer than this (already seen)
            'blacklist_radius': 0.6,
            'goal_timeout': 90.0,
            'replan_period': 5.0,         # re-check if the current frontier is still open
            'initial_spin': True,
            'spin_speed': 0.3,            # rad/s; slow turns keep SLAM scan matching locked
            'save_map': True,
            'map_save_path': os.path.expanduser('~/explored_map'),
        }
        for k, v in params.items():
            self.declare_parameter(k, v)

        self.map = None
        self.nav_status = 'IDLE'
        self.goal = None
        self.goal_frontier = None
        self.goal_time = None
        self.last_check = None
        self.blacklist = []
        self.done = False
        self.spin_until = None
        self.started = False
        self.empty_checks = 0
        self.final_look_done = False
        self.failed = []
        self.retried_failed = False

        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, latched)
        self.create_subscription(String, '/navigation_status', self.status_callback, latched)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.cancel_pub = self.create_publisher(Empty, '/cancel_goal', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/frontiers', 10)
        self.status_pub = self.create_publisher(String, '/exploration_status', latched)
        self.create_timer(1.0, self.step)
        self.create_timer(0.1, self.spin_step)
        self.publish_status('WAITING_FOR_MAP')
        self.get_logger().info('Frontier explorer started, waiting for SLAM map')

    def p(self, name):
        return self.get_parameter(name).value

    def publish_status(self, s):
        self.status_pub.publish(String(data=s))

    # ------------------------------------------------------------ inputs
    def map_callback(self, msg):
        self.map = msg

    def status_callback(self, msg):
        prev, self.nav_status = self.nav_status, msg.data
        if self.goal is None or prev == msg.data:
            return
        if msg.data == 'REACHED':
            self.get_logger().info('Frontier goal reached')
            self.blacklist.append(self.goal)
            self.goal = None
        elif msg.data == 'FAILED':
            self.get_logger().warn('Navigation failed, skipping this frontier for now')
            self.blacklist.append(self.goal)
            self.failed.append(self.goal)
            self.goal = None

    def robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform(self.p('global_frame'), self.p('base_frame'),
                                                rclpy.time.Time())
        except Exception:
            return None
        return t.transform.translation.x, t.transform.translation.y

    # ------------------------------------------------------------ spin
    def start_spin(self):
        turn_time = 2 * math.pi / self.p('spin_speed') + 0.5
        self.spin_until = self.get_clock().now() + Duration(seconds=turn_time)

    def spin_step(self):
        # One slow turn at the start so SLAM sees all around the robot
        if self.spin_until is None:
            return
        if self.get_clock().now() < self.spin_until:
            tw = Twist()
            tw.angular.z = self.p('spin_speed')
            self.cmd_pub.publish(tw)
        else:
            self.cmd_pub.publish(Twist())
            self.spin_until = None

    # ------------------------------------------------------------ main loop
    def step(self):
        if self.done or self.map is None or self.spin_until is not None:
            return
        pos = self.robot_xy()
        if pos is None:
            self.get_logger().warn('No map->base TF yet (is SLAM running?)', throttle_duration_sec=5.0)
            return
        if not self.started:
            self.started = True
            if self.p('initial_spin'):
                self.get_logger().info('Initial 360 deg scan')
                self.start_spin()
                return

        now = self.get_clock().now()
        if self.goal is not None:
            if (now - self.goal_time) > Duration(seconds=self.p('goal_timeout')):
                self.get_logger().warn('Frontier goal timed out, skipping it for now')
                self.blacklist.append(self.goal)
                self.failed.append(self.goal)
                self.cancel_pub.publish(Empty())
                self.goal = None
            elif (now - self.last_check) > Duration(seconds=self.p('replan_period')):
                self.last_check = now
                if not self.frontier_still_open():
                    self.get_logger().info('Current frontier already explored, choosing a new one')
                    self.goal = None
            if self.goal is not None:
                return

        target = self.choose_frontier(pos)
        if target is None:
            self.empty_checks += 1
            if self.empty_checks >= 3:
                if self.failed and not self.retried_failed:
                    # Give frontiers that failed earlier one more chance (the map
                    # has grown since, a path may exist now)
                    self.retried_failed = True
                    self.blacklist = [b for b in self.blacklist if b not in self.failed]
                    self.get_logger().info(f'Retrying {len(self.failed)} previously failed frontier(s)')
                    self.failed = []
                    self.empty_checks = 0
                    return
                if not self.final_look_done:
                    # One more look around before giving up: SLAM may not have
                    # integrated everything the robot could see yet
                    self.final_look_done = True
                    self.empty_checks = 0
                    self.get_logger().info('No frontiers left, final 360 deg look before finishing')
                    self.start_spin()
                    return
                self.finish()
            return
        self.empty_checks = 0
        self.final_look_done = False
        gx, gy, fx, fy, size = target
        self.send_goal(gx, gy, math.atan2(fy - gy, fx - gx))
        self.goal_frontier = (fx, fy)
        self.get_logger().info(
            f'Exploring frontier at ({fx:.2f}, {fy:.2f}), {size:.1f} m long -> goal ({gx:.2f}, {gy:.2f})')

    # ------------------------------------------------------------ frontiers
    def grids(self):
        info = self.map.info
        grid = np.array(self.map.data, dtype=np.int16).reshape(info.height, info.width)
        free = grid == 0
        unknown = grid < 0
        occupied = grid >= 50
        return info, free, unknown, occupied

    def frontier_still_open(self):
        if self.goal_frontier is None:
            return False
        info, free, unknown, _ = self.grids()
        fx, fy = self.goal_frontier
        cx = int((fx - info.origin.position.x) / info.resolution)
        cy = int((fy - info.origin.position.y) / info.resolution)
        r = int(0.5 / info.resolution)
        win = unknown[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1]
        return int(win.sum()) >= 5

    def choose_frontier(self, pos):
        info, free, unknown, occupied = self.grids()
        res = info.resolution
        ox, oy = info.origin.position.x, info.origin.position.y

        frontier = free & binary_dilation(unknown, EIGHT)
        labels, n = label(frontier, structure=EIGHT)
        if n == 0:
            return None

        # Cells where the robot fits (same rule as the A* planner: keep clear of real
        # obstacles; unknown cells are not traversable but do not push the robot away)
        clearance = distance_transform_edt(~occupied) * res
        safe = (clearance > self.p('robot_radius')) & free
        rx = int((pos[0] - ox) / res)
        ry = int((pos[1] - oy) / res)
        # Start the reachability flood from the nearest safe cell to the robot
        if not (0 <= rx < info.width and 0 <= ry < info.height) or not safe[ry, rx]:
            ys, xs = np.nonzero(safe)
            if len(xs) == 0:
                return None
            k = int(np.argmin((xs - rx) ** 2 + (ys - ry) ** 2))
            rx, ry = int(xs[k]), int(ys[k])
        travel = geodesic_distance(safe, (rx, ry)) * res

        reachable = np.isfinite(travel)
        sizes = np.bincount(labels.ravel())
        best, best_cost = None, math.inf
        for idx in range(1, n + 1):
            length = sizes[idx] * res
            if length < self.p('min_frontier_size'):
                continue
            cluster = labels == idx
            # Distance of every cell to this frontier (+ index of the nearest frontier cell)
            d_front, (ny, nx) = distance_transform_edt(~cluster, return_indices=True)
            d_front = d_front * res
            # Any reachable, safe cell close to ANY part of the frontier is a candidate goal
            cand = reachable & (d_front <= self.p('goal_search_radius')) & \
                (travel >= self.p('min_goal_dist')) & (clearance > self.p('goal_clearance'))
            if self.blacklist:
                ys_all, xs_all = np.nonzero(cand)
                wx, wy = ox + (xs_all + 0.5) * res, oy + (ys_all + 0.5) * res
                bad = np.zeros(len(xs_all), dtype=bool)
                for bx, by in self.blacklist:
                    bad |= np.hypot(wx - bx, wy - by) < self.p('blacklist_radius')
                cand[ys_all[bad], xs_all[bad]] = False
            if not cand.any():
                continue
            # Cheapest to reach, preferring spots near the frontier (better view)
            score = np.where(cand, travel + d_front, np.inf)
            gyi, gxi = np.unravel_index(int(np.argmin(score)), score.shape)
            dist = float(travel[gyi, gxi])
            cost = dist - self.p('info_gain_weight') * length
            if cost < best_cost:
                fxi, fyi = int(nx[gyi, gxi]), int(ny[gyi, gxi])
                best_cost = cost
                best = (ox + (gxi + 0.5) * res, oy + (gyi + 0.5) * res,
                        ox + (fxi + 0.5) * res, oy + (fyi + 0.5) * res, length)

        self.publish_markers(labels, n, sizes, best)
        return best

    # ------------------------------------------------------------ outputs
    def send_goal(self, x, y, yaw):
        g = PoseStamped()
        g.header.frame_id = self.p('global_frame')
        g.header.stamp = self.get_clock().now().to_msg()
        g.pose.position.x, g.pose.position.y = x, y
        g.pose.orientation.z, g.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.goal_pub.publish(g)
        self.goal = (x, y)
        self.goal_time = self.get_clock().now()
        self.last_check = self.goal_time
        self.publish_status('EXPLORING')

    def publish_markers(self, labels, n, sizes, best):
        info = self.map.info
        res, ox, oy = info.resolution, info.origin.position.x, info.origin.position.y
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        pts = Marker()
        pts.header.frame_id = self.p('global_frame')
        pts.ns, pts.id, pts.type = 'frontiers', 1, Marker.POINTS
        pts.scale.x = pts.scale.y = res
        pts.color.b, pts.color.g, pts.color.a = 1.0, 0.6, 1.0
        ys, xs = np.nonzero(labels > 0)
        pts.points = [Point(x=float(ox + (x + 0.5) * res), y=float(oy + (y + 0.5) * res), z=0.05)
                      for x, y in zip(xs[::2], ys[::2])]
        arr.markers.append(pts)
        if best is not None:
            m = Marker()
            m.header.frame_id = self.p('global_frame')
            m.ns, m.id, m.type = 'target', 2, Marker.SPHERE
            m.pose.position.x, m.pose.position.y, m.pose.position.z = best[2], best[3], 0.1
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.3
            m.color.r, m.color.g, m.color.a = 1.0, 0.3, 1.0
            arr.markers.append(m)
        self.marker_pub.publish(arr)

    def finish(self):
        self.done = True
        self.publish_status('COMPLETE')
        self.get_logger().info('Exploration complete: no reachable frontiers left')
        if self.p('save_map'):
            path = self.p('map_save_path')
            self.get_logger().info(f'Saving map to {path}.yaml / .pgm')
            try:
                subprocess.Popen(['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', path,
                                  '--ros-args', '-p', 'save_map_timeout:=10.0'])
            except Exception as e:
                self.get_logger().error(f'Could not start map saver: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
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
        # A second Ctrl+C (terminal + launch both send one) must not print a traceback
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, ExternalShutdownException, Exception):
            pass


if __name__ == '__main__':
    main()
