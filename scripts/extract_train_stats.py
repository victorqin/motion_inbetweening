import os
import sys
import argparse


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
package_root = os.path.join(project_root, "packages")
sys.path.append(package_root)


from motion_inbetween.config import load_config_by_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract stats for training. For debugging purposes only. "
        "Training model directly will trigger stats extraction.")
    parser.add_argument("model", choices=["context", "detail", "rmi"])
    parser.add_argument("config", help="model config name")
    args = parser.parse_args()

    config = load_config_by_name(args.config)
    config.save_to_workspace()

    if args.model == "context":
        from motion_inbetween.train import context_model
        context_model.get_train_stats(config, use_cache=False)
    elif args.model == "detail":
        from motion_inbetween.train import detail_model
        detail_model.get_train_stats(config, use_cache=False)
    elif args.model == "rmi":
        from motion_inbetween.train import rmi
        rmi.get_train_stats(config, use_cache=False)
        rmi.get_rmi_benchmark_stats(config, use_cache=False)
