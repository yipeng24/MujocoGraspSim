#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Listen to two TransformStamped topics (env/tgt) and on Ctrl-C write poses.
It can also load obj->vicon transforms from config and save:
  1) vicon->world (raw)
  2) obj->world (composed)
"""

import ast
import math
import os
import threading

import rospy
from geometry_msgs.msg import TransformStamped


_pose_lock = threading.Lock()
_env_pose = None
_tgt_pose = None


def _pose_cb_env(msg: TransformStamped):
    global _env_pose
    with _pose_lock:
        _env_pose = msg


def _pose_cb_tgt(msg: TransformStamped):
    global _tgt_pose
    with _pose_lock:
        _tgt_pose = msg


def _load_simple_yaml(path: str):
    data = {}
    if not path or not os.path.exists(path):
        return data
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "#" in line:
                    line = line.split("#", 1)[0].strip()
                    if not line:
                        continue
                if ":" not in line:
                    continue
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                if not key:
                    continue
                if val == "":
                    data[key] = None
                    continue
                try:
                    data[key] = ast.literal_eval(val)
                    continue
                except Exception:
                    pass
                vlow = val.lower()
                if vlow in ("true", "false"):
                    data[key] = (vlow == "true")
                    continue
                try:
                    data[key] = int(val)
                    continue
                except Exception:
                    pass
                try:
                    data[key] = float(val)
                    continue
                except Exception:
                    pass
                data[key] = val
    except Exception as e:
        rospy.logwarn("[pcd_pose_to_yaml] Failed to read %s: %s", path, str(e))
    return data


def _fmt_scalar(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        s = f"{v:.8f}"
        s = s.rstrip("0").rstrip(".")
        return s if s else "0"
    return str(v)


def _fmt_value(v):
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt_scalar(x) for x in v) + "]"
    return _fmt_scalar(v)


def _write_simple_yaml(path: str, data: dict):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        keys_first = ["transform_enable", "transform_xyz", "transform_quat_xyzw"]
        lines = []
        for k in keys_first:
            if k in data:
                lines.append(f"{k}: {_fmt_value(data[k])}")
        for k in sorted(data.keys()):
            if k in keys_first:
                continue
            lines.append(f"{k}: {_fmt_value(data[k])}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        rospy.loginfo("[pcd_pose_to_yaml] Wrote %s", path)
    except Exception as e:
        rospy.logerr("[pcd_pose_to_yaml] Failed to write %s: %s", path, str(e))


def _normalize_quat(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    return x / n, y / n, z / n, w / n


def _quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return _normalize_quat(x, y, z, w)


def _rotate_vec(q, v):
    x, y, z, w = q
    # q * v * q^-1 with v as pure quaternion
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return rx, ry, rz


def _compose_transform(t1, q1, t2, q2):
    # T1: a->b (t1,q1), T2: b->c (t2,q2) => T: a->c
    q = _quat_multiply(q1, q2)
    r_t2 = _rotate_vec(q1, t2)
    t = (t1[0] + r_t2[0], t1[1] + r_t2[1], t1[2] + r_t2[2])
    return t, q


def _get_transform_from_config(path: str):
    data = _load_simple_yaml(path)
    t = data.get("transform_xyz", [0.0, 0.0, 0.0])
    q = data.get("transform_quat_xyzw", [0.0, 0.0, 0.0, 1.0])
    try:
        t = [float(t[0]), float(t[1]), float(t[2])]
    except Exception:
        t = [0.0, 0.0, 0.0]
    try:
        q = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
    except Exception:
        q = [0.0, 0.0, 0.0, 1.0]
    q = _normalize_quat(q[0], q[1], q[2], q[3])
    return tuple(t), tuple(q)


def _update_config(path: str, pose: TransformStamped, set_enable: bool):
    data = _load_simple_yaml(path)
    if set_enable:
        data["transform_enable"] = 1
    if pose is None:
        rospy.logwarn("[pcd_pose_to_yaml] No pose for %s, keeping existing values", path)
        _write_simple_yaml(path, data)
        return

    t = pose.transform.translation
    q = pose.transform.rotation
    qx, qy, qz, qw = _normalize_quat(q.x, q.y, q.z, q.w)

    data["transform_xyz"] = [float(t.x), float(t.y), float(t.z)]
    data["transform_quat_xyzw"] = [float(qx), float(qy), float(qz), float(qw)]

    # Remove RPY if present to avoid ambiguity
    if "transform_rpy_deg" in data:
        data.pop("transform_rpy_deg")

    _write_simple_yaml(path, data)


def _update_config_from_transform(path: str, t, q, set_enable: bool):
    data = _load_simple_yaml(path)
    if set_enable:
        data["transform_enable"] = 1
    data["transform_xyz"] = [float(t[0]), float(t[1]), float(t[2])]
    data["transform_quat_xyzw"] = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
    if "transform_rpy_deg" in data:
        data.pop("transform_rpy_deg")
    _write_simple_yaml(path, data)


def _on_shutdown(env_path, tgt_path, env_vicon_path, tgt_vicon_path,
                 env_obj_to_vicon_path, tgt_obj_to_vicon_path, set_enable):
    with _pose_lock:
        env_pose = _env_pose
        tgt_pose = _tgt_pose

    # Raw vicon->world
    _update_config(env_vicon_path, env_pose, set_enable)
    _update_config(tgt_vicon_path, tgt_pose, set_enable)

    # Compose obj->world = (vicon->world) * (obj->vicon)
    env_obj_t, env_obj_q = _get_transform_from_config(env_obj_to_vicon_path)
    tgt_obj_t, tgt_obj_q = _get_transform_from_config(tgt_obj_to_vicon_path)

    if env_pose is not None:
        t = env_pose.transform.translation
        q = env_pose.transform.rotation
        vicon_t = (float(t.x), float(t.y), float(t.z))
        vicon_q = _normalize_quat(q.x, q.y, q.z, q.w)
        obj_t, obj_q = _compose_transform(vicon_t, vicon_q, env_obj_t, env_obj_q)
        _update_config_from_transform(env_path, obj_t, obj_q, set_enable)
    else:
        rospy.logwarn("[pcd_pose_to_yaml] No env pose; obj->world not updated.")

    if tgt_pose is not None:
        t = tgt_pose.transform.translation
        q = tgt_pose.transform.rotation
        vicon_t = (float(t.x), float(t.y), float(t.z))
        vicon_q = _normalize_quat(q.x, q.y, q.z, q.w)
        obj_t, obj_q = _compose_transform(vicon_t, vicon_q, tgt_obj_t, tgt_obj_q)
        _update_config_from_transform(tgt_path, obj_t, obj_q, set_enable)
    else:
        rospy.logwarn("[pcd_pose_to_yaml] No tgt pose; obj->world not updated.")


def main():
    rospy.init_node("pcd_pose_to_yaml", anonymous=True)

    env_pose_topic = rospy.get_param("~env_pose_topic", "/vicon/cube/cube")
    tgt_pose_topic = rospy.get_param("~tgt_pose_topic", "/vicon/cup/cup")
    env_config_path = rospy.get_param("~env_config_path", "")
    tgt_config_path = rospy.get_param("~tgt_config_path", "")
    set_enable = bool(rospy.get_param("~set_transform_enable", False))

    env_vicon_config_path = rospy.get_param("~env_vicon_config_path", "")
    tgt_vicon_config_path = rospy.get_param("~tgt_vicon_config_path", "")
    env_obj_to_vicon_config_path = rospy.get_param("~env_obj_to_vicon_config_path", "")
    tgt_obj_to_vicon_config_path = rospy.get_param("~tgt_obj_to_vicon_config_path", "")

    if not env_config_path or not tgt_config_path:
        rospy.logerr("[pcd_pose_to_yaml] env_config_path or tgt_config_path is empty")
        return
    if not env_vicon_config_path or not tgt_vicon_config_path:
        rospy.logerr("[pcd_pose_to_yaml] env_vicon_config_path or tgt_vicon_config_path is empty")
        return
    if not env_obj_to_vicon_config_path or not tgt_obj_to_vicon_config_path:
        rospy.logerr("[pcd_pose_to_yaml] env_obj_to_vicon_config_path or tgt_obj_to_vicon_config_path is empty")
        return

    rospy.Subscriber(env_pose_topic, TransformStamped, _pose_cb_env, queue_size=1)
    rospy.Subscriber(tgt_pose_topic, TransformStamped, _pose_cb_tgt, queue_size=1)

    rospy.loginfo("[pcd_pose_to_yaml] Listening env: %s", env_pose_topic)
    rospy.loginfo("[pcd_pose_to_yaml] Listening tgt: %s", tgt_pose_topic)
    rospy.loginfo("[pcd_pose_to_yaml] Will write env obj->world: %s", env_config_path)
    rospy.loginfo("[pcd_pose_to_yaml] Will write tgt obj->world: %s", tgt_config_path)
    rospy.loginfo("[pcd_pose_to_yaml] Will write env vicon->world: %s", env_vicon_config_path)
    rospy.loginfo("[pcd_pose_to_yaml] Will write tgt vicon->world: %s", tgt_vicon_config_path)
    rospy.loginfo("[pcd_pose_to_yaml] Using env obj->vicon: %s", env_obj_to_vicon_config_path)
    rospy.loginfo("[pcd_pose_to_yaml] Using tgt obj->vicon: %s", tgt_obj_to_vicon_config_path)

    rospy.on_shutdown(lambda: _on_shutdown(env_config_path, tgt_config_path,
                                           env_vicon_config_path, tgt_vicon_config_path,
                                           env_obj_to_vicon_config_path, tgt_obj_to_vicon_config_path,
                                           set_enable))
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
