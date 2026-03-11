import argparse
import mujoco
from mujoco_viewer import MujocoViewer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="Path to MJCF xml")
    ap.add_argument("--steps", type=int, default=0, help="0 means run forever")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)

    viewer = MujocoViewer(model, data)

    i = 0
    try:
        while viewer.is_alive:
            mujoco.mj_step(model, data)
            viewer.render()
            i += 1
            if args.steps > 0 and i >= args.steps:
                break
    finally:
        try:
            viewer.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
