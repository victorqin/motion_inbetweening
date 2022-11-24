import os
import sys
import argparse


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
package_root = os.path.join(project_root, "packages")
sys.path.append(package_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train detail model.")
    parser.add_argument("det_config", help="detail config name")
    parser.add_argument("ctx_config", help="context config name")
    args = parser.parse_args()

    from motion_inbetween.train import detail_model
    from motion_inbetween.config import load_config_by_name

    det_config = load_config_by_name(args.det_config)
    det_config.save_to_workspace()
    ctx_config = load_config_by_name(args.ctx_config)
    ctx_config.save_to_workspace(det_config["workspace"])

    detail_model.train(det_config, ctx_config)
