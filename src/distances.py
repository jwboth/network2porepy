"""Helper methods to determine distances."""

import numpy as np


def point_line_dist(pt: np.ndarray, seg: np.ndarray) -> float:
    a, b = seg
    ab = b - a
    ap = pt - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    closest = a + t * ab
    return float(np.linalg.norm(pt - closest))


def line_distance(line1: np.ndarray, line2: np.ndarray) -> float:
    """Calculate minimum distance between two line segments."""

    dists = [
        point_line_dist(line1[0], line2),
        point_line_dist(line1[1], line2),
        point_line_dist(line2[0], line1),
        point_line_dist(line2[1], line1),
    ]
    return min(dists)


def cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    """2D scalar cross product."""
    return a[0] * b[1] - a[1] * b[0]


def segment_intersection_point(
    line1: np.ndarray,
    line2: np.ndarray,
    tol: float,
) -> np.ndarray | None:
    """Compute a unique segment intersection point, if it exists."""
    p = line1[0]
    r = line1[1] - line1[0]
    q = line2[0]
    s = line2[1] - line2[0]
    rxs = cross_2d(r, s)
    q_p = q - p
    q_pxr = cross_2d(q_p, r)

    if abs(rxs) <= tol:
        # Parallel (including collinear) treated as no unique intersection point.
        return None

    t = cross_2d(q_p, s) / rxs
    u = q_pxr / rxs
    if -tol <= t <= 1.0 + tol and -tol <= u <= 1.0 + tol:
        return p + np.clip(t, 0.0, 1.0) * r
    return None


def closest_extension_point(
    base_point: np.ndarray, point: np.ndarray, line: np.ndarray
) -> np.ndarray:
    """Find the closest point on a line for extension."""
    a, b = line
    ab = b - a

    def det(alpha):
        v1 = a - base_point + alpha * ab
        v2 = a - point + alpha * ab
        return np.linalg.det(np.array([v1, v2]))

    alpha = 0.0
    det_alpha = det(alpha)
    sign_det_alpha = np.sign(det_alpha)
    inc = 0.01

    while True:
        alpha += inc
        if alpha > 1:
            return a + alpha * ab
        if np.sign(det(alpha)) != sign_det_alpha:
            alpha -= inc
            inc /= 2
            if inc < 1e-8:
                return a + alpha * ab
