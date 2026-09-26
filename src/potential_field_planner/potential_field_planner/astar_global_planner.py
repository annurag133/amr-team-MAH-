import heapq
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
from scipy.ndimage import distance_transform_edt

import tf2_ros
from tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan

from potential_field_planner.scan_utils import planar_transform, project_scan
from std_msgs.msg import Empty, String

SQRT2 = math.sqrt(2.0)
NEIGHBORS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2)]


class AStarGlobalPlanner(Node):

    def __init__(self):
        super().__init__('astar_global_planner')

        self.declare_parameter('robot_radius', 0.30)
        self.declare_parameter('safety_margin', 0.35)
        self.declare_parameter('wall_cost_weight', 5.0)
        self.declare_parameter('waypoint_spacing', 0.4)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('unknown_is_free', False)
        self.declare_parameter('max_snap_dist', 0.6)
        self.declare_parameter('new_obstacle_min_dist', 0.3)
        self.declare_parameter('use_laser_obstacles', True)
        self.declare_parameter('check_path_blocked', False)

        self.map = None
        self.free = None
        self.cost = None
        self.goal = None

        latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(OccupancyGrid, '/map', self.map_callback, latched)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.create_subscription(Empty, '/replan', self.replan_callback, 10)
        self.create_subscription(LaserScan, '/scan', lambda m: setattr(self, 'scan', m),
                                 qos_profile_sensor_data)
        self.scan = None
        self.dynamic = None
        self.path_cells = None
        self.nav_status = 'IDLE'
        self.create_subscription(String, '/navigation_status',
                                 lambda m: setattr(self, 'nav_status', m.data), latched)
        self.create_timer(1.0, self.check_path_blocked)
        self.path_pub = self.create_publisher(Path, '/global_path', latched)

        self.get_logger().info('A* Global Planner started')

    def p(self, name):
        return self.get_parameter(name).value

    def map_callback(self, msg):
        self.map = msg
        info = msg.info
        grid = np.array(msg.data, dtype=np.int16).reshape(info.height, info.width)
        # Only real obstacles push paths away; unknown cells just can't be driven
        # through (otherwise nothing near a frontier or a doorway is ever reachable).
        self.occupied = grid >= 50
        unknown = grid < 0
        self.untraversable = np.zeros_like(unknown) if self.p('unknown_is_free') else unknown
        self.obstacle = self.occupied | unknown
        # Distance to the nearest mapped wall/unknown cell; laser points close to
        # mapped walls are just the walls seen with small localisation error.
        self.static_dist = distance_transform_edt(~self.obstacle) * info.resolution
        self.dynamic = np.zeros_like(self.obstacle)
        self.path_cells = None
        self.rebuild()

        self.get_logger().info(
            f'Map received: {info.width} x {info.height}, '
            f'resolution={info.resolution:.3f}, free cells={int(self.free.sum())}',
            throttle_duration_sec=30.0)

    def rebuild(self):
        # Distance (m) from every cell to the nearest obstacle (mapped or laser-seen)
        dist = distance_transform_edt(~(self.occupied | self.dynamic)) * self.map.info.resolution
        self.free = (dist > self.p('robot_radius')) & ~self.untraversable
        margin = self.p('safety_margin')
        # Extra cost for cells within the safety margin keeps paths centred
        self.cost = 1.0 + self.p('wall_cost_weight') * np.clip(
            1.0 - (dist - self.p('robot_radius')) / max(margin, 1e-3), 0.0, 1.0)

    def goal_callback(self, msg):
        frame = self.p('global_frame')
        if msg.header.frame_id and msg.header.frame_id != frame:
            try:
                t = self.tf_buffer.lookup_transform(
                    frame, msg.header.frame_id, rclpy.time.Time(),
                    timeout=Duration(seconds=0.5))
                msg = PoseStamped(header=t.header,
                                  pose=do_transform_pose(msg.pose, t))
            except Exception as e:
                self.get_logger().error(f'Cannot transform goal to {frame}: {e}')
                return
        self.goal = msg
        if self.map is not None:
            self.dynamic[:] = False
            self.rebuild()
        self.get_logger().info(
            f'New goal: x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}')
        self.plan_path()

    def replan_callback(self, _):
        if self.goal is None or self.map is None:
            return
        n = self.merge_scan_obstacles()
        self.get_logger().warn(f'Replan requested ({n} new laser obstacle cells)')
        self.plan_path()

    def scan_cells(self):
        # Grid cells hit by the current laser scan (within 4 m), in the map frame
        scan = self.scan
        if scan is None or self.map is None:
            return None
        try:
            t = self.tf_buffer.lookup_transform(
                self.p('global_frame'), scan.header.frame_id, rclpy.time.Time(),
                timeout=Duration(seconds=0.2))
        except Exception:
            return None
        r = np.asarray(scan.ranges, dtype=float)
        a = scan.angle_min + np.arange(len(r)) * scan.angle_increment
        ok = np.isfinite(r) & (r > scan.range_min) & (r < min(scan.range_max, 4.0))
        x, y = project_scan(r[ok], a[ok], planar_transform(t))
        info = self.map.info
        gx = np.floor((x - info.origin.position.x) / info.resolution).astype(int)
        gy = np.floor((y - info.origin.position.y) / info.resolution).astype(int)
        inside = (gx >= 0) & (gx < info.width) & (gy >= 0) & (gy < info.height)
        gx, gy = gx[inside], gy[inside]
        new = self.static_dist[gy, gx] > self.p('new_obstacle_min_dist')
        return gx[new], gy[new]

    def merge_scan_obstacles(self):
        # Obstacles the laser sees but the static map lacks (people, boxes) are
        # remembered until the next goal, so replans route around them consistently.
        if not self.p('use_laser_obstacles'):
            return 0
        cells = self.scan_cells()
        if cells is None:
            return 0
        gx, gy = cells
        before = int(self.dynamic.sum())
        self.dynamic[gy, gx] = True
        self.dynamic &= ~self.obstacle
        added = int(self.dynamic.sum()) - before
        if added:
            self.rebuild()
        return added

    def check_path_blocked(self):
        # Replan as soon as a laser obstacle sits on the upcoming path
        if not (self.p('use_laser_obstacles') and self.p('check_path_blocked')) \
                or self.goal is None or self.path_cells is None \
                or len(self.path_cells) == 0 \
                or self.nav_status != 'FOLLOWING':
            return
        pos = self.robot_position()
        if pos is None:
            return
        rx, ry = self.world_to_grid(*pos)
        path = np.asarray(self.path_cells, dtype=float)
        i0 = int(np.argmin(((path - (rx, ry)) ** 2).sum(axis=1)))
        path = path[i0:]
        cells = self.scan_cells()
        if cells is None:
            return
        gx, gy = cells
        new = ~self.obstacle[gy, gx] & ~self.dynamic[gy, gx]
        if not np.any(new):
            return
        pts = np.column_stack([gx[new], gy[new]]).astype(float)
        r = self.p('robot_radius') / self.map.info.resolution
        d2 = ((path[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
        if np.any(d2 < r * r):
            n = self.merge_scan_obstacles()
            self.get_logger().warn(f'Path blocked by an obstacle not in the map ({n} cells), replanning')
            self.plan_path()

    def robot_position(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.p('global_frame'), self.p('base_frame'), rclpy.time.Time(),
                timeout=Duration(seconds=0.5))
        except Exception as e:
            self.get_logger().error(
                f'No {self.p("global_frame")}->{self.p("base_frame")} TF '
                f'(is localisation running and initialised?): {e}')
            return None
        return t.transform.translation.x, t.transform.translation.y

    def world_to_grid(self, x, y):
        info = self.map.info
        return (int(math.floor((x - info.origin.position.x) / info.resolution)),
                int(math.floor((y - info.origin.position.y) / info.resolution)))

    def grid_to_world(self, gx, gy):
        info = self.map.info
        return (info.origin.position.x + (gx + 0.5) * info.resolution,
                info.origin.position.y + (gy + 0.5) * info.resolution)

    def in_bounds(self, x, y):
        return 0 <= x < self.map.info.width and 0 <= y < self.map.info.height

    def is_free(self, x, y):
        return self.in_bounds(x, y) and bool(self.free[y, x])

    def nearest_free(self, cell):
        if self.is_free(*cell):
            return cell
        max_r = int(self.p('max_snap_dist') / self.map.info.resolution)
        best, best_d = None, None
        for r in range(1, max_r + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    c = (cell[0] + dx, cell[1] + dy)
                    d = dx * dx + dy * dy
                    if self.is_free(*c) and (best_d is None or d < best_d):
                        best, best_d = c, d
            if best is not None:
                return best
        return None

    def astar(self, start, goal):
        open_set = [(0.0, 0.0, start)]
        came_from = {}
        g_score = {start: 0.0}
        closed = set()
        gx, gy = goal

        while open_set:
            _, g, current = heapq.heappop(open_set)
            if current in closed:
                continue
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            closed.add(current)

            cx, cy = current
            for dx, dy, step in NEIGHBORS:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in closed or not self.is_free(nx, ny):
                    continue
                # No diagonal corner-cutting through blocked cells
                if dx and dy and not (self.is_free(cx + dx, cy) and self.is_free(cx, cy + dy)):
                    continue
                tentative = g + step * self.cost[ny, nx]
                if tentative < g_score.get((nx, ny), math.inf):
                    g_score[(nx, ny)] = tentative
                    came_from[(nx, ny)] = current
                    h = math.hypot(gx - nx, gy - ny)
                    heapq.heappush(open_set, (tentative + h, tentative, (nx, ny)))
        return None

    def line_free(self, a, b):
        n = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        for i in range(n + 1):
            t = i / n if n else 0.0
            x = int(round(a[0] + (b[0] - a[0]) * t))
            y = int(round(a[1] + (b[1] - a[1]) * t))
            if not self.is_free(x, y):
                return False
        return True

    def sparsify(self, path):
        # Keep waypoints no further apart than waypoint_spacing, and never
        # skip a stretch that isn't straight-line free.
        max_cells = max(1, int(self.p('waypoint_spacing') / self.map.info.resolution))
        out = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = min(i + max_cells, len(path) - 1)
            while j > i + 1 and not self.line_free(path[i], path[j]):
                j -= 1
            out.append(path[j])
            i = j
        return out

    def publish_path(self, cells, exact_goal=False):
        if not cells:
            self.path_cells = None
        msg = Path()
        msg.header.frame_id = self.p('global_frame')
        msg.header.stamp = self.get_clock().now().to_msg()
        for idx, (gx, gy) in enumerate(cells):
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x, pose.pose.position.y = self.grid_to_world(gx, gy)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        if msg.poses:
            if exact_goal:
                msg.poses[-1].pose.position = self.goal.pose.position
            msg.poses[-1].pose.orientation = self.goal.pose.orientation
        self.path_pub.publish(msg)

    def plan_path(self):
        if self.map is None:
            self.get_logger().warn('No map yet, cannot plan')
            return
        pos = self.robot_position()
        if pos is None:
            return

        raw_start = self.world_to_grid(*pos)
        raw_goal = self.world_to_grid(self.goal.pose.position.x,
                                      self.goal.pose.position.y)
        if not self.in_bounds(*raw_goal):
            self.get_logger().error('Goal is outside the map')
            self.publish_path([])
            return

        start = self.nearest_free(raw_start)
        goal = self.nearest_free(raw_goal)
        if start is None:
            self.get_logger().error('Robot is inside an obstacle (check localisation)')
            self.publish_path([])
            return
        if goal is None:
            self.get_logger().error('Goal is in or too close to an obstacle')
            self.publish_path([])
            return

        t0 = self.get_clock().now()
        grid_path = self.astar(start, goal)
        if grid_path is None and self.dynamic is not None and self.dynamic.any():
            self.get_logger().warn('No path with laser obstacles; retrying on the plain map')
            self.dynamic[:] = False
            self.rebuild()
            start, goal = self.nearest_free(raw_start), self.nearest_free(raw_goal)
            if start is not None and goal is not None:
                grid_path = self.astar(start, goal)
        ms = (self.get_clock().now() - t0).nanoseconds / 1e6
        if grid_path is None:
            self.get_logger().error(f'A* found no path ({ms:.0f} ms)')
            self.publish_path([])
            return

        self.path_cells = grid_path
        cells = self.sparsify(grid_path)
        if goal != raw_goal:
            self.get_logger().warn('Goal too close to an obstacle, moved to nearest safe cell')
        self.publish_path(cells, exact_goal=(goal == raw_goal))
        self.get_logger().info(
            f'Path: {len(grid_path)} cells -> {len(cells)} waypoints ({ms:.0f} ms)')


def main(args=None):
    rclpy.init(args=args)
    node = AStarGlobalPlanner()
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
