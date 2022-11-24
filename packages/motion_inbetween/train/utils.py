import os
import math
import json
import glob

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from motion_inbetween.data import loader
from motion_inbetween.data import utils_torch as data_utils


def load_checkpoint(config, model, optimizer=None, scheduler=None, suffix=""):
    checkpoint_path = config["train"]["checkpoint"] + suffix

    if os.path.exists(checkpoint_path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        epoch = checkpoint["epoch"]
        iteration = checkpoint["iteration"] + 1
        model.load_state_dict(checkpoint["model"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler:
            scheduler.load_state_dict(checkpoint["scheduler"])
        print("Load checkpoint from {}. Current iteration: {}.".format(
            checkpoint_path, checkpoint["iteration"]))
    else:
        epoch = 1
        iteration = 1
        print("No checkpoint found at {}. Skip.".format(checkpoint_path))

    return epoch, iteration


def save_checkpoint(config, model, epoch, iteration, optimizer,
                    scheduler=None, suffix="", n_ckp=1):
    checkpoint_path = config["train"]["checkpoint"] + suffix
    checkpoint = {
        "epoch": epoch,
        "iteration": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scheduler:
        checkpoint["scheduler"] = scheduler.state_dict()

    if n_ckp > 1:
        # keep previously saved checkpoint
        existing_ckps = glob.glob(checkpoint_path)
        existing_ckps.extend(glob.glob(checkpoint_path+".*"))

        ckp_path_data = []
        for ckp_path in existing_ckps:
            version_str = ckp_path.replace(checkpoint_path, "")
            version_str = version_str.strip().lstrip(".")
            if version_str == "":
                version_num = 0
                new_path = ckp_path + ".1"
            else:
                try:
                    version_num = int(version_str)
                    new_path = ckp_path.rsplit(
                        ".", 1)[0] + "." + str(version_num+1)
                except ValueError:
                    import traceback
                    traceback.print_exc()
                    print("Invalid checkpoint path: {}".format(ckp_path))

            if version_num + 1 < n_ckp:
                ckp_path_data.append((version_num, ckp_path, new_path))

        # rename from the biggest version to the smallest version
        ckp_path_data.sort()
        ckp_path_data.reverse()
        for _, old_path, new_path in ckp_path_data:
            os.replace(old_path, new_path)

    torch.save(checkpoint, checkpoint_path)
    print("Save checkpoint to {}.".format(checkpoint_path))


def init_bvh_dataset(config, dataset_name, device,
                     shuffle=True, dtype=torch.float32):
    dataset = loader.BvhDataSet(**config["datasets"][dataset_name],
                                device=device, dtype=dtype)
    print("{} clips in dataset.".format(len(dataset)))
    data_loader = DataLoader(dataset, batch_size=config["train"]["batch_size"],
                             shuffle=shuffle)
    return dataset, data_loader


def get_val_datasets(config, device,
                     exclude=("train", "train_stats",
                              "bench_stats", "benchmark"),
                     shuffle=False, dtype=torch.float32):
    dataset_names = []
    datasets = []
    dataset_loaders = []

    for dataset_name in config["datasets"]:
        if dataset_name in exclude:
            continue
        else:
            ds, dl = init_bvh_dataset(
                config, dataset_name, device, shuffle, dtype)
            dataset_names.append(dataset_name)
            datasets.append(ds)
            dataset_loaders.append(dl)
    return dataset_names, datasets, dataset_loaders


def get_noam_lr_scheduler(config, optimizer):
    warm_iters = config["train"]["lr_warmup_iterations"]

    def _lr_lambda(iteration, warm_iters=warm_iters):
        # return a multiplier instead of learning rate.
        if iteration < warm_iters:
            return iteration * warm_iters ** -1.5
        else:
            return 1.0 / math.sqrt(iteration)

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)


def cal_r_loss(x, y, seq_slice, indices, weights=None):
    dim_slice = slice(indices["r_start_idx"], indices["r_end_idx"])

    # l1 loss
    delta = x[..., seq_slice, dim_slice] - y[..., seq_slice, dim_slice]
    if weights is not None:
        delta = delta * weights[..., None]
    return torch.mean(torch.abs(delta))


def cal_c_loss(c_gt, c_out, seq_slice, weights=None):
    # l1 loss
    delta = c_gt[..., seq_slice, :] - c_out[..., seq_slice, :]
    if weights is not None:
        delta = delta * weights[..., None]
    return torch.mean(torch.abs(delta))


def cal_smooth_loss(new_global_positions, seq_slice, weights=None):
    seq_slice1 = slice(seq_slice.start, seq_slice.stop + 1)
    seq_slice2 = slice(seq_slice.start - 1, seq_slice.stop)

    # l1 loss
    delta = (new_global_positions[..., seq_slice1, :, :] -
             new_global_positions[..., seq_slice2, :, :])

    if weights is not None:
        delta = delta * weights[..., None, None]

    return torch.mean(torch.abs(delta))


def cal_p_loss(global_positions, new_global_positions, seq_slice,
               weights=None):

    # l1 loss
    delta = (global_positions[..., seq_slice, :, :] -
             new_global_positions[..., seq_slice, :, :])
    if weights is not None:
        delta = delta * weights[..., None, None]

    return torch.mean(torch.abs(delta))


def cal_f_loss(gpos_out, c_out, seq_slice):
    # FIXME: hard-coded foot indices
    # shape: (..., seq, joint, 3)
    foot_vel = data_utils.extract_foot_vel(
        gpos_out, foot_joint_idx=(3, 4, 7, 8))

    # sever gradient back propagation on c_out,
    # otherwise c_out will tend to be zero
    delta = (
        c_out[..., seq_slice, :, None].detach() *
        foot_vel[..., seq_slice, :, :]
    )

    # l1 loss
    return torch.mean(torch.abs(delta))


def get_new_positions(positions, y, indices, seq_slice=slice(None, None)):
    p_slice = slice(indices["p_start_idx"], indices["p_end_idx"])

    positions_new = positions.clone()
    positions_new[..., seq_slice, 0, :] = y[..., seq_slice, p_slice]

    return positions_new


def get_new_rotations(y, indices, rotations=None, seq_slice=slice(None, None)):
    r_slice = slice(indices["r_start_idx"], indices["r_end_idx"])

    if rotations is None:
        rot = y[..., r_slice]
        rot = rot.reshape(*rot.shape[:-1], -1, 6)
        rot = data_utils.matrix6D_to_9D_torch(rot)
    else:
        rot_tmp = y[..., seq_slice, r_slice]
        rot_tmp = rot_tmp.reshape(*rot_tmp.shape[:-1], -1, 6)
        rot_tmp = data_utils.matrix6D_to_9D_torch(rot_tmp)

        rot = rotations.clone()
        rot[..., seq_slice, :, :, :] = rot_tmp

    return rot


def anim_post_process(data, data_gt, seq_slice):
    # An approximate method to find an offset applied on output curve
    # which minimize the animation curve curvature at
    # last context frame and target frame.
    # Here we linearly extrapolate from both directions to obtain
    # start_extrapolated and end_extrapolated.
    # Then we minimize the distance between predicted value and
    # extrapolated value.
    start_idx = seq_slice.start
    end_idx = seq_slice.stop

    start_extrapolated = (
        2 * data_gt[..., start_idx - 1, :] - data_gt[..., start_idx - 2, :])
    start_value = data[..., start_idx, :]

    # Note: in order to make this work, we must provide one
    # extra frame after target frame so that we can properly
    # extrapolate. (This can be viewed as providing velocity
    # at the target frame.)
    end_extrapolated = (
        2 * data_gt[..., end_idx, :] - data_gt[..., end_idx + 1, :])
    end_value = data[..., end_idx - 1, :]

    delta = (start_extrapolated - start_value +
             end_extrapolated - end_value) / 2

    data[..., seq_slice, :] = (
        data[..., seq_slice, :] + delta[..., None, :])

    return data


# Visdom Visualization ######################################################
def get_next_data_idx(vis, window):
    existing_data = vis.get_window_data(window)
    if existing_data:
        d = json.loads(existing_data)
        info_idx = d["content"]["data"][0]["x"][-1] + 1
    else:
        info_idx = 1
    return info_idx


def init_visdom(config):
    import visdom
    vis = visdom.Visdom(env=config["visdom"]["env"])
    info_idx = get_next_data_idx(vis, "loss")
    return vis, info_idx


def to_visdom(vis, info_idx, contents):
    """
    Send data to Visdom for display.
    """
    for win, label, value in contents:
        vis.line(
            X=torch.FloatTensor([info_idx]),
            Y=torch.FloatTensor([value]),
            win=win, update="append", name=label,
            opts={
                "title": win,
                "showlegend": True
            })
