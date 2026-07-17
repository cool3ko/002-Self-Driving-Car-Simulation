"""Geometry helpers for collision and ray-casting."""


def line_intersect(a, b, c, d):
    """Find the intersection of segment a-b with segment c-d.

    Args:
        a, b: Endpoints of the first segment, each an (x, y) tuple.
        c, d: Endpoints of the second segment, each an (x, y) tuple.

    Returns:
        tuple: (x, y, t) of the intersection point and how far along
        a-b it occurred (0 = at a, 1 = at b), or None if the segments
        don't cross.
    """
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d

    denominator = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
    if denominator == 0:
        return None

    t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / denominator
    u = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / denominator

    if 0 <= t <= 1 and 0 <= u <= 1:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1), t)
    return None


def polys_intersect(poly1, poly2):
    """Check whether two polygons (or segments) overlap.

    Args:
        poly1: List of (x, y) points forming a closed polygon (or a
            single 2-point segment).
        poly2: Same as poly1.

    Returns:
        bool: True if any edge of poly1 crosses any edge of poly2.
    """
    for i in range(len(poly1)):
        a = poly1[i]
        b = poly1[(i + 1) % len(poly1)]
        for j in range(len(poly2)):
            c = poly2[j]
            d = poly2[(j + 1) % len(poly2)]
            if line_intersect(a, b, c, d):
                return True
    return False
