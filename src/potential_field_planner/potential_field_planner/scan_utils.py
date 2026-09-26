import numpy as np


def planar_transform(transform):
    """2D projection of a full 3D TF transform (handles lasers mounted upside down).

    Returns (tx, ty, a00, a01, a10, a11) so that a point (u, v, 0) in the source
    frame maps to (tx + a00*u + a01*v, ty + a10*u + a11*v) in the target frame.
    """
    q = transform.transform.rotation
    t = transform.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    return (t.x, t.y,
            1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
            2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z))


def project_scan(ranges, angles, m):
    """Laser beams (range, angle in the laser frame) -> x, y in the target frame."""
    tx, ty, a00, a01, a10, a11 = m
    u = ranges * np.cos(angles)
    v = ranges * np.sin(angles)
    return tx + a00 * u + a01 * v, ty + a10 * u + a11 * v
