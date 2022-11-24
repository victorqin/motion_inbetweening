import os
import sys
import argparse


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
package_root = os.path.join(project_root, "packages")
sys.path.append(package_root)


import torch

from motion_inbetween import benchmark
from motion_inbetween.train import rmi
from motion_inbetween.train import utils as train_utils
from motion_inbetween.config import load_config_by_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline benchmark.")
    parser.add_argument("config", help="config name")
    args = parser.parse_args()

    config = load_config_by_name(args.config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset, data_loader = train_utils.init_bvh_dataset(
        config, "benchmark", device, shuffle=False, dtype=torch.float64)

    mean_rmi, std_rmi = rmi.get_rmi_benchmark_stats_torch(
        config, dataset.dtype, device)

    trans_lens = [5, 15, 30, 45]
    context_len = config["train"]["context_len"]

    for trans_len in trans_lens:
        target_idx = context_len + trans_len

        zerov_pos, zerov_quat, zerov_npss = benchmark.get_zerov_loss(
            dataset, data_loader, context_len, target_idx, mean_rmi, std_rmi)

        interp_pos, interp_quat, interp_npss = benchmark.get_interp_loss(
            dataset, data_loader, context_len, target_idx, mean_rmi, std_rmi)

        print(
            "trans: {:2d} \n"
            "zerov_pos: {:.4f}, zerov_quat: {:.2f}, zerov_npss: {:.4f}\n"
            "inter_pos: {:.4f}, inter_quat: {:.2f}, inter_npss: {:.4f}".format(
                trans_len, zerov_pos, zerov_quat, zerov_npss,
                interp_pos, interp_quat, interp_npss)
        )
