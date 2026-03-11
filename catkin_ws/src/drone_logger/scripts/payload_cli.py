#!/usr/bin/env python3
"""
payload_cli.py
==============
交互式命令行工具，方便在运行时修改 payload 的质量和质心位置。
等效于手动调用 rosservice call /drone/set_payload。

使用：
  rosrun drone_logger payload_cli.py
  rosrun drone_logger payload_cli.py --mass 0.2 --com 0.01 0 0
  rosrun drone_logger payload_cli.py --remove
"""

import sys
import argparse
import rospy
from drone_logger.srv import SetPayload


def call_set_payload(mass, com_x, com_y, com_z, enable):
    rospy.wait_for_service('/drone/set_payload', timeout=5.0)
    svc = rospy.ServiceProxy('/drone/set_payload', SetPayload)
    resp = svc(mass=mass, com_x=com_x, com_y=com_y, com_z=com_z, enable=enable)
    return resp


def main():
    parser = argparse.ArgumentParser(description='Payload parameter CLI')
    parser.add_argument('--mass', type=float, default=None, help='Payload mass [kg]')
    parser.add_argument('--com', type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        metavar=('X', 'Y', 'Z'), help='COM offset in gripper frame [m]')
    parser.add_argument('--remove', action='store_true', help='Remove payload (set mass=0)')
    args = parser.parse_args()

    rospy.init_node('payload_cli', anonymous=True)

    if args.remove:
        resp = call_set_payload(0.0, 0.0, 0.0, 0.0, enable=False)
        print(f"[payload_cli] Remove: {resp.message}  total_mass={resp.total_mass:.3f} kg")
        return

    if args.mass is None:
        # Interactive mode
        print("=== Payload CLI (interactive) ===")
        print("Commands: set <mass> [com_x com_y com_z] | remove | quit")
        while not rospy.is_shutdown():
            try:
                line = input("payload> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            parts = line.split()
            cmd = parts[0].lower()
            if cmd in ('q', 'quit', 'exit'):
                break
            elif cmd == 'remove':
                resp = call_set_payload(0.0, 0.0, 0.0, 0.0, enable=False)
                print(f"  -> {resp.message}  total_mass={resp.total_mass:.3f} kg")
            elif cmd == 'set' and len(parts) >= 2:
                m = float(parts[1])
                cx = float(parts[2]) if len(parts) > 2 else 0.0
                cy = float(parts[3]) if len(parts) > 3 else 0.0
                cz = float(parts[4]) if len(parts) > 4 else 0.0
                resp = call_set_payload(m, cx, cy, cz, enable=True)
                print(f"  -> {resp.message}  total_mass={resp.total_mass:.3f} kg")
            else:
                print("  Usage: set <mass> [com_x com_y com_z] | remove | quit")
    else:
        cx, cy, cz = args.com
        resp = call_set_payload(args.mass, cx, cy, cz, enable=True)
        print(f"[payload_cli] {resp.message}  total_mass={resp.total_mass:.3f} kg")


if __name__ == '__main__':
    main()
