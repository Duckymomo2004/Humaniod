"""Geometry helpers for diagnostic support polygons; imports USD only on demand."""

import numpy as np


def hull2d(points):
    p = sorted(set(map(tuple, np.asarray(points, dtype=float).reshape(-1, 2))))
    if len(p) <= 2:
        return np.asarray(p).reshape(-1, 2)

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower, upper = [], []
    for v in p:
        while len(lower) > 1 and cross(lower[-2], lower[-1], v) <= 1e-12:
            lower.pop()
        lower.append(v)
    for v in reversed(p):
        while len(upper) > 1 and cross(upper[-2], upper[-1], v) <= 1e-12:
            upper.pop()
        upper.append(v)
    return np.asarray(lower[:-1] + upper[:-1])


def support_metrics(points, com_xy):
    hull = hull2d(points)
    if len(hull) < 3:
        return hull, float("nan"), 0.0
    edge = np.roll(hull, -1, axis=0) - hull
    delta = np.asarray(com_xy) - hull
    signed = (edge[:, 0]*delta[:, 1] - edge[:, 1]*delta[:, 0]) / np.linalg.norm(edge, axis=1)
    area = .5 * abs(np.sum(hull[:, 0]*np.roll(hull[:, 1], -1) - hull[:, 1]*np.roll(hull[:, 0], -1)))
    return hull, float(signed.min()), float(area)


def collision_vertices(stage, body_path):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(body_path)
    cache = UsdGeom.XformCache()
    inverse = cache.GetLocalToWorldTransform(root).GetInverse()
    shapes, vertices = [], []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if enabled is False:
            continue
        kind = prim.GetTypeName()
        transform = cache.GetLocalToWorldTransform(prim) * inverse
        if kind == "Mesh":
            local = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        elif kind == "Cube":
            half = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
            local = np.array([[x, y, z] for x in [-half, half] for y in [-half, half] for z in [-half, half]])
        elif kind == "Sphere":
            r = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
            # Six extrema identify the collision envelope; actual support uses
            # raw contact points, never a spherical bounding box as a patch.
            local = np.vstack([np.eye(3), -np.eye(3)]) * r
        else:
            raise ValueError(f"Unsupported collision shape for this diagnostic: {kind} {prim.GetPath()}")
        body_local = np.array([transform.Transform(Gf.Vec3d(*map(float, p))) for p in local])
        vertices.append(body_local)
        shapes.append({"path": str(prim.GetPath()), "type": kind, "vertex_count": len(local),
                       "body_local_min": body_local.min(0).tolist(), "body_local_max": body_local.max(0).tolist(),
                       "approximation": str(prim.GetAttribute("physics:approximation").Get())})
    if not vertices:
        raise RuntimeError(f"No enabled collision geometry in {body_path}")
    return np.concatenate(vertices), shapes


def transform_vertices(local, pos, quat):
    # PhysX / current Lab use quaternion xyzw.
    v = np.asarray(local)
    xyz, w = np.asarray(quat[:3]), float(quat[3])
    return v + 2 * np.cross(xyz, np.cross(xyz, v) + w*v) + pos
