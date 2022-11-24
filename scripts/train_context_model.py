import os
import sys
import argparse


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
package_root = os.path.join(project_root, "packages")
sys.path.append(package_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train context model.")
    parser.add_argument("config", help="config name")
    args = parser.parse_args()

    from motion_inbetween.train import context_model
    from motion_inbetween.config import load_config_by_name

    config = load_config_by_name(args.config)
    config.save_to_workspace()
    context_model.train(config)
