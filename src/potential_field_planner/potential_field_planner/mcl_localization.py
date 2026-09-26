"""Monte Carlo Localisation (particle filter) for the Robile.

Drop-in replacement for nav2 AMCL: consumes /map, /scan, /initialpose and the
odom->base TF, publishes map->odom TF, /mcl_pose and /particle_cloud.
"""
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
from geometry_msgs.msg import (Pose, PoseWithCovarianceStamped,
                               TransformStamped)
from nav2_msgs.msg import Particle, ParticleCloud
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from potential_field_planner.scan_utils import planar_transform, project_scan


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ParticleFilter:
    """Pure numpy MCL core (no ROS), so it can be tested offline."""

    def __init__(self, n=500, alphas=(0.2, 0.2, 0.2, 0.2), sigma_hit=0.2,
                 z_hit=0.95, z_rand=0.05, max_dist=2.0, rng=None):
        self.n = n
        self.a1, self.a2, self.a3, self.a4 = alphas
        self.sigma_hit = sigma_hit
        self.z_hit = z_hit
        self.z_rand = z_rand
        self.max_dist = max_dist
        self.rng = rng or np.random.default_rng()
        self.p = np.zeros((0, 3))
        self.w = np.zeros(0)
        self.field = None

    # ---------------------------------------------------------------- map
    def set_map(self, grid, resolution, origin_x, origin_y, range_max=10.0):
        """grid: (h, w) int array, 0..100 occupancy, -1 unknown."""
        self.res = resolution
        self.ox, self.oy = origin_x, origin_y
        self.h, self.w_cells = grid.shape
        occ = grid >= 65
        dist = distance_transform_edt(~occ) * resolution
        dist = np.minimum(dist, self.max_dist)
        # Likelihood field: Gaussian around obstacles plus uniform noise
        self.field = self.z_hit * np.exp(-dist ** 2 / (2 * self.sigma_hit ** 2)) \
            + self.z_rand / range_max
        self.outside = self.z_rand / range_max
        self.free_cells = np.argwhere(grid == 0)

    # ------------------------------------------------------------ init
    def init_gaussian(self, x, y, yaw, sx=0.3, sy=0.3, syaw=0.2):
        self.p = np.column_stack([
            self.rng.normal(x, sx, self.n),
            self.rng.normal(y, sy, self.n),
            wrap(self.rng.normal(yaw, syaw, self.n))])
        self.w = np.full(self.n, 1.0 / self.n)

    def init_uniform(self, n=None):
        n = n or self.n * 4
        idx = self.rng.integers(0, len(self.free_cells), n)
        cells = self.free_cells[idx]
        self.p = np.column_stack([
            self.ox + (cells[:, 1] + self.rng.random(n)) * self.res,
            self.oy + (cells[:, 0] + self.rng.random(n)) * self.res,
            self.rng.uniform(-np.pi, np.pi, n)])
        self.w = np.full(n, 1.0 / n)

    # ------------------------------------------------------------ predict
    def predict(self, dx, dy, dth):
        """(dx, dy, dth): odometry motion expressed in the previous robot frame."""
        m = len(self.p)
        trans = math.hypot(dx, dy)
        rot = abs(dth)
        s_xy = self.a3 * trans + self.a4 * rot
        s_th = self.a1 * rot + self.a2 * trans
        ndx = dx + self.rng.normal(0.0, s_xy + 1e-9, m)
        ndy = dy + self.rng.normal(0.0, s_xy + 1e-9, m)
        ndth = dth + self.rng.normal(0.0, s_th + 1e-9, m)
        c, s = np.cos(self.p[:, 2]), np.sin(self.p[:, 2])
        self.p[:, 0] += c * ndx - s * ndy
        self.p[:, 1] += s * ndx + c * ndy
        self.p[:, 2] = wrap(self.p[:, 2] + ndth)

    # ------------------------------------------------------------ update
    def update(self, bx, by):
        """bx, by: laser endpoints already expressed in the robot base frame."""
        px, py, pth = self.p[:, 0:1], self.p[:, 1:2], self.p[:, 2:3]
        c, s = np.cos(pth), np.sin(pth)
        ex = px + c * bx[None, :] - s * by[None, :]
        ey = py + s * bx[None, :] + c * by[None, :]
        gx = np.floor((ex - self.ox) / self.res).astype(np.int64)
        gy = np.floor((ey - self.oy) / self.res).astype(np.int64)
        inside = (gx >= 0) & (gx < self.w_cells) & (gy >= 0) & (gy < self.h)
        pz = np.full(gx.shape, self.outside)
        pz[inside] = self.field[gy[inside], gx[inside]]
        # AMCL-style combination: robust to a few bad beams
        like = 1.0 + np.sum(pz ** 3, axis=1)
        self.w = self.w * like
        total = self.w.sum()
        if not np.isfinite(total) or total <= 0:
            self.w = np.full(len(self.p), 1.0 / len(self.p))
        else:
            self.w /= total

    def n_eff(self):
        return 1.0 / np.sum(self.w ** 2)

    # ------------------------------------------------------------ resample
    def resample(self):
        """Low-variance (systematic) resampling down/up to self.n particles."""
        # Stay at the large (global) particle count until the cloud has converged,
        # otherwise it collapses early onto a wrong, look-alike place.
        x, y = self.p[:, 0], self.p[:, 1]
        spread = math.sqrt(np.average((x - np.average(x, weights=self.w)) ** 2, weights=self.w) +
                           np.average((y - np.average(y, weights=self.w)) ** 2, weights=self.w))
        n = self.n if spread < 0.5 else max(self.n, len(self.p))
        positions = (self.rng.random() + np.arange(n)) / n
        cum = np.cumsum(self.w)
        cum[-1] = 1.0
        idx = np.searchsorted(cum, positions)
        self.p = self.p[idx].copy()
        self.w = np.full(n, 1.0 / n)

    # ------------------------------------------------------------ estimate
    def estimate(self):
        w = self.w
        x = float(np.sum(w * self.p[:, 0]))
        y = float(np.sum(w * self.p[:, 1]))
        yaw = math.atan2(float(np.sum(w * np.sin(self.p[:, 2]))),
                         float(np.sum(w * np.cos(self.p[:, 2]))))
        vx = float(np.sum(w * (self.p[:, 0] - x) ** 2))
        vy = float(np.sum(w * (self.p[:, 1] - y) ** 2))
        vyaw = float(np.sum(w * wrap(self.p[:, 2] - yaw) ** 2))
        return x, y, yaw, vx, vy, vyaw


class MCLNode(Node):

    def __init__(self):
        super().__init__('mcl_localization')
        params = {
            'num_particles': 500,
            'global_particles': 3000,
            'alpha1': 0.2, 'alpha2': 0.2, 'alpha3': 0.2, 'alpha4': 0.2,
            'sigma_hit': 0.2, 'z_hit': 0.95, 'z_rand': 0.05,
            'likelihood_max_dist': 2.0,
            'max_beams': 60,
            'update_min_d': 0.05,
            'update_min_a': 0.05,
            'resample_threshold': 0.5,
            'transform_tolerance': 0.2,
            'global_frame': 'map',
            'odom_frame': 'odom',
            'base_frame': 'base_footprint',
            'set_initial_pose': True,
            'initial_x': 0.0, 'initial_y': 0.0, 'initial_yaw': 0.0,
        }
        for k, v in params.items():
            self.declare_parameter(k, v)

        self.pf = ParticleFilter(
            n=self.p('num_particles'),
            alphas=(self.p('alpha1'), self.p('alpha2'), self.p('alpha3'), self.p('alpha4')),
            sigma_hit=self.p('sigma_hit'), z_hit=self.p('z_hit'), z_rand=self.p('z_rand'),
            max_dist=self.p('likelihood_max_dist'))

        self.map_ready = False
        self.initialized = False
        self.last_odom = None
        self.force_update = True
        self.laser_pose = None
        self.map_to_odom = None

        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.create_subscription(OccupancyGrid, '/map', self.map_callback, latched)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose',
                                 self.initialpose_callback, 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/mcl_pose', latched)
        self.cloud_pub = self.create_publisher(ParticleCloud, '/particle_cloud',
                                               qos_profile_sensor_data)
        self.create_service(Empty, '/reinitialize_global_localization', self.global_callback)
        # Keep map->odom alive between scans so downstream TF lookups never go stale
        self.create_timer(0.1, self.republish_tf)

        self.get_logger().info('MCL localisation started, waiting for map')

    def p(self, name):
        return self.get_parameter(name).value

    # ------------------------------------------------------------ callbacks
    def map_callback(self, msg):
        info = msg.info
        grid = np.array(msg.data, dtype=np.int16).reshape(info.height, info.width)
        self.pf.set_map(grid, info.resolution, info.origin.position.x, info.origin.position.y)
        self.map_ready = True
        self.get_logger().info(f'Map received: {info.width} x {info.height}')
        if not self.initialized and self.p('set_initial_pose'):
            self.pf.init_gaussian(self.p('initial_x'), self.p('initial_y'), self.p('initial_yaw'))
            self.start()
            self.get_logger().info(
                f'Initial pose ({self.p("initial_x"):.2f}, {self.p("initial_y"):.2f}, '
                f'{self.p("initial_yaw"):.2f})')

    def initialpose_callback(self, msg):
        if not self.map_ready:
            self.get_logger().warn('Initial pose ignored: no map yet')
            return
        pose = msg.pose.pose
        cov = msg.pose.covariance
        sx = max(math.sqrt(max(cov[0], 0.0)), 0.1)
        sy = max(math.sqrt(max(cov[7], 0.0)), 0.1)
        syaw = max(math.sqrt(max(cov[35], 0.0)), 0.1)
        self.pf.init_gaussian(pose.position.x, pose.position.y, yaw_of(pose.orientation),
                              sx, sy, syaw)
        self.start()
        self.get_logger().info(
            f'Initial pose set: ({pose.position.x:.2f}, {pose.position.y:.2f}, '
            f'{yaw_of(pose.orientation):.2f})')

    def global_callback(self, request, response):
        if self.map_ready:
            self.pf.init_uniform(self.p('global_particles'))
            self.start()
            self.get_logger().info('Global localisation: particles spread over the free map')
        return response

    def start(self):
        self.initialized = True
        self.last_odom = None
        self.force_update = True
        self.publish_cloud()

    # ------------------------------------------------------------ helpers
    def lookup(self, target, source, stamp):
        try:
            return self.tf_buffer.lookup_transform(target, source, stamp,
                                                   timeout=Duration(seconds=0.05))
        except Exception:
            try:
                return self.tf_buffer.lookup_transform(target, source, rclpy.time.Time())
            except Exception:
                return None

    @staticmethod
    def xyyaw(t):
        tr = t.transform.translation
        return tr.x, tr.y, yaw_of(t.transform.rotation)

    def scan_callback(self, scan):
        if not (self.map_ready and self.initialized):
            return
        stamp = rclpy.time.Time.from_msg(scan.header.stamp)

        if self.laser_pose is None:
            t = self.lookup(self.p('base_frame'), scan.header.frame_id, rclpy.time.Time())
            if t is None:
                self.get_logger().warn('No base->laser TF yet', throttle_duration_sec=5.0)
                return
            self.laser_pose = planar_transform(t)

        t = self.lookup(self.p('odom_frame'), self.p('base_frame'), stamp)
        if t is None:
            self.get_logger().warn('No odom->base TF', throttle_duration_sec=5.0)
            return
        odom = self.xyyaw(t)

        if self.last_odom is not None:
            x0, y0, th0 = self.last_odom
            ddx, ddy = odom[0] - x0, odom[1] - y0
            dx = math.cos(th0) * ddx + math.sin(th0) * ddy
            dy = -math.sin(th0) * ddx + math.cos(th0) * ddy
            dth = float(wrap(odom[2] - th0))
            moved = math.hypot(dx, dy) >= self.p('update_min_d') or \
                abs(dth) >= self.p('update_min_a')
        else:
            dx = dy = dth = 0.0
            moved = False

        if moved or self.force_update:
            if self.last_odom is not None:
                self.pf.predict(dx, dy, dth)
            self.last_odom = odom
            self.force_update = False
            self.measurement_update(scan)
            if self.pf.n_eff() < self.p('resample_threshold') * len(self.pf.p) \
                    or len(self.pf.p) != self.pf.n:
                self.pf.resample()
            self.publish_cloud()
            self.publish_pose(scan.header.stamp)
            self.compute_map_to_odom(odom)

        self.publish_tf(scan.header.stamp)

    def measurement_update(self, scan):
        r = np.asarray(scan.ranges, dtype=float)
        a = scan.angle_min + np.arange(len(r)) * scan.angle_increment
        valid = np.isfinite(r) & (r > scan.range_min) & (r < scan.range_max)
        r, a = r[valid], a[valid]
        if r.size == 0:
            return
        k = self.p('max_beams')
        if r.size > k:
            idx = np.linspace(0, r.size - 1, k).astype(int)
            r, a = r[idx], a[idx]
        bx, by = project_scan(r, a, self.laser_pose)
        self.pf.update(bx, by)

    def compute_map_to_odom(self, odom):
        x, y, yaw = self.pf.estimate()[:3]
        ox, oy, oyaw = odom
        # map_T_odom = map_T_base * inverse(odom_T_base)
        tyaw = yaw - oyaw
        c, s = math.cos(tyaw), math.sin(tyaw)
        tx = x - (c * ox - s * oy)
        ty = y - (s * ox + c * oy)
        self.map_to_odom = (tx, ty, tyaw)

    def publish_tf(self, stamp_msg):
        if self.map_to_odom is None:
            return
        stamp = rclpy.time.Time.from_msg(stamp_msg) + \
            Duration(seconds=self.p('transform_tolerance'))
        self.last_tf_stamp = stamp
        tx, ty, tyaw = self.map_to_odom
        t = TransformStamped()
        t.header.stamp = stamp.to_msg()
        t.header.frame_id = self.p('global_frame')
        t.child_frame_id = self.p('odom_frame')
        t.transform.translation.x = tx
        t.transform.translation.y = ty
        t.transform.rotation.z = math.sin(tyaw / 2.0)
        t.transform.rotation.w = math.cos(tyaw / 2.0)
        self.tf_broadcaster.sendTransform(t)

    def republish_tf(self):
        if self.map_to_odom is None:
            return
        # Only fill gaps; scan-driven publishes carry the proper sensor time
        now = self.get_clock().now()
        last = getattr(self, 'last_tf_stamp', None)
        if last is None or (now - last) > Duration(seconds=0.3):
            self.publish_tf(now.to_msg())

    def publish_pose(self, stamp):
        x, y, yaw, vx, vy, vyaw = self.pf.estimate()
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.p('global_frame')
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        cov = [0.0] * 36
        cov[0], cov[7], cov[35] = vx, vy, vyaw
        msg.pose.covariance = cov
        self.pose_pub.publish(msg)

    def publish_cloud(self):
        msg = ParticleCloud()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.p('global_frame')
        for (x, y, th), w in zip(self.pf.p, self.pf.w):
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.orientation.z = math.sin(th / 2.0)
            pose.orientation.w = math.cos(th / 2.0)
            msg.particles.append(Particle(pose=pose, weight=float(w)))
        self.cloud_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MCLNode()
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
