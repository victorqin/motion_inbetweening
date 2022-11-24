import os
import sys
import json
import argparse


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
package_root = os.path.join(project_root, "packages")
sys.path.append(package_root)


import torch

from motion_inbetween import benchmark, visualization
from motion_inbetween.model import ContextTransformer
from motion_inbetween.config import load_config_by_name
from motion_inbetween.train import rmi
from motion_inbetween.train import context_model
from motion_inbetween.train import utils as train_utils
from motion_inbetween.data import utils_torch as data_utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate context model. "
                                     "Post-processing is applied by default.")
    parser.add_argument("config", help="config name")
    parser.add_argument("-s", "--dataset",
                        help="dataset name (default=benchmark)",
                        default="benchmark")
    parser.add_argument("-i", "--index", type=int, help="data index")
    parser.add_argument("-t", "--trans", type=int, default=30,
                        help="transition length (default=30)")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="debug mode")
    parser.add_argument("-p", "--post_processing", action="store_true",
                        default=False, help="apply post-processing")
    args = parser.parse_args()

    config = load_config_by_name(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset, data_loader = train_utils.init_bvh_dataset(
        config, args.dataset, device=device, shuffle=False)

    # initialize model
    model = ContextTransformer(config["model"]).to(device)
    # load checkpoint
    epoch, iteration = train_utils.load_checkpoint(config, model)

    if args.index is None:
        res = context_model.eval_on_dataset(
            config, data_loader, model, args.trans, args.debug,
            args.post_processing)

        if args.debug:
            gpos_loss, gquat_loss, npss_loss, loss_data = res

            json_path = "{}_{}_{}_ranking.json".format(
                args.config, args.dataset, args.trans)
            with open(json_path, "w") as fh:
                json.dump(loss_data, fh)
        else:
            gpos_loss, gquat_loss, npss_loss = res

        print(config["name"])
        print("trans {}: gpos: {:.4f}, gquat: {:.4f}, npss: {:.4f}{}".format(
            args.trans, gpos_loss, gquat_loss, npss_loss,
            " (w/ post-processing)" if args.post_processing else ""))
    else:
        indices = config["indices"]
        context_len = config["train"]["context_len"]
        target_idx = context_len + args.trans
        seq_slice = slice(context_len, target_idx)
        window_len = context_len + args.trans + 2
        dtype = dataset.dtype

        # attention mask
        atten_mask = context_model.get_attention_mask(
            window_len, context_len, target_idx, device)

        mean, std = context_model.get_train_stats_torch(
            config, dtype, device)
        mean_rmi, std_rmi = rmi.get_rmi_benchmark_stats_torch(
            config, dataset.dtype, device)

        data = dataset[args.index]
        (positions, rotations, global_positions, global_rotations,
            foot_contact, parents, data_idx) = data

        positions = positions[..., :window_len, :, :]
        rotations = rotations[..., :window_len, :, :, :]
        global_positions = global_positions[..., :window_len, :, :]
        global_rotations = global_rotations[..., :window_len, :, :, :]
        foot_contact = foot_contact[..., :window_len, :]

        positions = positions.unsqueeze(0)
        rotations = rotations.unsqueeze(0)
        positions, rotations = data_utils.to_start_centered_data(
            positions, rotations, context_len)

        pos_new, rot_new = context_model.evaluate(
            model, positions, rotations, seq_slice, indices,
            mean, std, atten_mask, args.post_processing)

        gpos_batch_loss, gquat_batch_loss, _, _ = \
            benchmark.get_rmi_style_batch_loss(
                positions, rotations, pos_new, rot_new, parents,
                context_len, target_idx, mean_rmi, std_rmi)

        print(config["name"])
        print("trans: {}, idx: {}, gpos: {:.4f}, gquat: {:.4f}{}".format(
            args.trans, args.index, gpos_batch_loss[0], gquat_batch_loss[0],
            " (w/ post-processing)" if args.post_processing else ""))

        json_path_gt = "./{}_{}_{}_{}_gt.json".format(
            args.config, args.dataset, args.trans, args.index)
        visualization.save_data_to_json(
            json_path_gt, positions[0], rotations[0],
            foot_contact, parents)

        json_path = "./{}_{}_{}_{}.json".format(
            args.config, args.dataset, args.trans, args.index)
        visualization.save_data_to_json(
            json_path, pos_new[0], rot_new[0],
            foot_contact, parents)

        if args.debug:
            print("\nDEBUG:")
            pos_interp, rot_interp = benchmark.get_interpolated_local_pos_rot(
                positions, rotations, seq_slice)
            gpos_interp_loss, gquat_interp_loss, _, _ = \
                benchmark.get_rmi_style_batch_loss(
                    positions, rotations, pos_interp, rot_interp, parents,
                    context_len, target_idx, mean_rmi, std_rmi)
            print("interp: gpos: {:.4f}, gquat: {:.4f}".format(
                gpos_interp_loss[0], gquat_interp_loss[0]))

            json_path_inter = "./{}_{}_{}_{}_inter.json".format(
                args.config, args.dataset, args.trans, args.index)
            visualization.save_data_to_json(
                json_path_inter, pos_interp[0], rot_interp[0],
                foot_contact, parents)
