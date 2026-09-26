import math
import threading
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Pose
from PIL import Image


def logit(p):
    return math.log(p / (1.0 - p))


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class OccupancyGridMapper(Node):
    def __init__(self):
        super().__init__("occupancy_grid_mapper")

        # ---------------- Parameters ----------------
        self.map_size_m = 20.0
        self.resolution = 0.05

        self.p_prior = 0.5
        self.p_occ = 0.8
        self.p_free = 0.3

        self.free_threshold = 0.35
        self.occupied_threshold = 0.65

        self.log_min = -5.0
        self.log_max = 5.0

        self.update_dist_thresh = 0.05
        self.update_angle_thresh = 0.05

        self.hit_tolerance = 0.08
        self.map_filename = "closed_walls_map.png"

        # ---------------- Map setup ----------------
        self.width = int(self.map_size_m / self.resolution)
        self.height = int(self.map_size_m / self.resolution)

        self.origin_x = -self.map_size_m / 2.0
        self.origin_y = -self.map_size_m / 2.0

        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)

        self.l_prior = logit(self.p_prior)
        self.l_occ = logit(self.p_occ)
        self.l_free = logit(self.p_free)

        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None

        self.last_update_x = None
        self.last_update_y = None
        self.last_update_yaw = None

        self.map_pub = self.create_publisher(OccupancyGrid, "/map", 10)

        self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)

        self.timer = self.create_timer(0.5, self.publish_map)

        self.running = True

        keyboard_thread = threading.Thread(target=self.keyboard_listener)
        keyboard_thread.daemon = True
        keyboard_thread.start()

        self.get_logger().info("Occupancy grid mapper started. Press ENTER to stop and save map.")

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

    def world_to_grid(self, x, y):
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)

        if 0 <= gx < self.width and 0 <= gy < self.height:
            return gx, gy

        return None

    def should_update(self):
        if self.last_update_x is None:
            return True

        dx = self.robot_x - self.last_update_x
        dy = self.robot_y - self.last_update_y
        dist = math.sqrt(dx * dx + dy * dy)

        dtheta = abs(math.atan2(
            math.sin(self.robot_yaw - self.last_update_yaw),
            math.cos(self.robot_yaw - self.last_update_yaw)
        ))

        return dist > self.update_dist_thresh or dtheta > self.update_angle_thresh

    def bresenham(self, x0, y0, x1, y1):
        cells = []

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        err = dx - dy

        x, y = x0, y0

        while True:
            cells.append((x, y))

            if x == x1 and y == y1:
                break

            e2 = 2 * err

            if e2 > -dy:
                err -= dy
                x += sx

            if e2 < dx:
                err += dx
                y += sy

        return cells

    def scan_callback(self, scan):
        if self.robot_x is None or self.robot_y is None or self.robot_yaw is None:
            return

        if not self.should_update():
            return

        robot_cell = self.world_to_grid(self.robot_x, self.robot_y)

        if robot_cell is None:
            return

        rx, ry = robot_cell

        angle = scan.angle_min

        for r in scan.ranges:
            if math.isinf(r) or math.isnan(r):
                angle += scan.angle_increment
                continue

            r_clamped = min(r, scan.range_max)

            global_angle = self.robot_yaw + angle

            end_x = self.robot_x + r_clamped * math.cos(global_angle)
            end_y = self.robot_y + r_clamped * math.sin(global_angle)

            end_cell = self.world_to_grid(end_x, end_y)

            if end_cell is None:
                angle += scan.angle_increment
                continue

            ex, ey = end_cell

            ray_cells = self.bresenham(rx, ry, ex, ey)

            # Mark free cells first
            for cx, cy in ray_cells[:-1]:
                self.log_odds[cy, cx] += self.l_free - self.l_prior
                self.log_odds[cy, cx] = np.clip(
                    self.log_odds[cy, cx],
                    self.log_min,
                    self.log_max
                )

            # Mark occupied cell only if laser actually hit something
            if r < scan.range_max - self.hit_tolerance:
                ox, oy = ray_cells[-1]
                self.log_odds[oy, ox] += self.l_occ - self.l_prior
                self.log_odds[oy, ox] = np.clip(
                    self.log_odds[oy, ox],
                    self.log_min,
                    self.log_max
                )

            angle += scan.angle_increment

        self.last_update_x = self.robot_x
        self.last_update_y = self.robot_y
        self.last_update_yaw = self.robot_yaw

        self.publish_map()

    def log_odds_to_prob(self):
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))

    def publish_map(self):
        msg = OccupancyGrid()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height

        msg.info.origin = Pose()
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        probs = self.log_odds_to_prob()

        data = []

        for y in range(self.height):
            for x in range(self.width):
                p = probs[y, x]

                if p > self.occupied_threshold:
                    data.append(100)
                elif p < self.free_threshold:
                    data.append(0)
                else:
                    data.append(-1)

        msg.data = data
        self.map_pub.publish(msg)

    def save_png(self):
        probs = self.log_odds_to_prob()

        img = np.zeros((self.height, self.width), dtype=np.uint8)

        for y in range(self.height):
            for x in range(self.width):
                p = probs[y, x]

                if p > self.occupied_threshold:
                    img[y, x] = 0          # occupied = black
                elif p < self.free_threshold:
                    img[y, x] = 255        # free = white
                else:
                    img[y, x] = 127        # unknown = gray

        img = np.flipud(img)

        image = Image.fromarray(img)
        image.save(self.map_filename)

        self.get_logger().info(f"Map saved as {self.map_filename}")

    def keyboard_listener(self):
        input()
        self.get_logger().info("Keyboard interruption received. Saving map...")
        self.running = False
        self.save_png()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    node = OccupancyGridMapper()

    try:
        while rclpy.ok() and node.running:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C received. Saving map...")
        node.save_png()
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
