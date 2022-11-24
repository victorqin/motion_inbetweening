import os
import pickle
import random

import numpy as np

import torch
from torch.optim import Adam

from motion_inbetween import benchmark
from motion_inbetween.data import utils_torch as data_utils
from motion_inbetween.train import utils as train_utils

from motion_inbetween.model.rmi import (
    RmiMotionDiscriminator, RmiMotionGenerator,
    TargetNoiseEmbedding, TimeToArrivalEmbedding
)


def get_rmi_benchmark_stats(config, use_cache=True,
                            stats_folder=None, dataset_name="bench_stats"):
    """
    Get benchmark stats (mean and std of global position) following
    Robust Motion In-betweening(Harvey et al., 2020) paper.

    Args:
        config (dict): config
        use_cache (bool, optional): Use cached stats. Defaults to True.
        stats_folder (str, optional): The folder where cached stats is
            located. If None, the bvh folder of dataset_name in config is used.
            Defaults to None.
        dataset_name (str, optional): name of the dataset in config which is
            used to calculate benchmark stats. Defaults to "bench_stats".

    Returns:
        tuple: (gpos mean , gpos std)
    """
    context_len = config["train"]["context_len"]
    dataset_config = config["datasets"][dataset_name]

    if stats_folder is None:
        stats_folder = dataset_config["bvh_folder"]
    stats_path = os.path.join(
        stats_folder, "bench_stats_{}_{}_{}.pkl".format(
            context_len, dataset_config["window"], dataset_config["offset"],
            dataset_config["start_frame"]))

    if use_cache and os.path.exists(stats_path):
        with open(stats_path, "rb") as fh:
            train_stats = pickle.load(fh)
        print("RMI style benchmark stats load from {}".format(stats_path))
    else:
        print("Calculating RMI style benchmark stats ...")
        # calculate training stats
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset, data_loader = train_utils.init_bvh_dataset(
            config, dataset_name, device, shuffle=False, dtype=torch.float64)

        global_pos = []
        for i, data in enumerate(data_loader, 0):
            (positions, rotations, global_positions, global_rotations,
             foot_contact, parents, data_idx) = data
            parents = parents[0]

            positions, rotations = data_utils.to_mean_centered_data(
                positions, rotations, context_len)
            global_rotations, global_positions = data_utils.fk_torch(
                rotations, positions, parents)
            global_pos.append(global_positions.cpu().numpy())

        # (batch, seq, joint, 3)
        global_pos = np.concatenate(global_pos, axis=0)
        # (batch, seq, joint*3)
        global_pos = global_pos.reshape(*global_pos.shape[0:2], -1)

        mean = np.mean(global_pos, axis=(0, 1))
        std = np.std(global_pos, axis=(0, 1))

        train_stats = {
            "mean": mean,
            "std": std
        }

        with open(stats_path, "wb") as fh:
            pickle.dump(train_stats, fh)

        print("RMI style benchmark stats wrote to {}".format(stats_path))

    return train_stats["mean"], train_stats["std"]


def get_rmi_benchmark_stats_torch(config, dtype, device,
                                  use_cache=True, stats_folder=None,
                                  dataset_name="bench_stats"):
    mean, std = get_rmi_benchmark_stats(config, use_cache,
                                        stats_folder, dataset_name)
    mean = mean.reshape(*mean.shape[:-1], -1, 3)
    std = std.reshape(*std.shape[:-1], -1, 3)
    mean = torch.tensor(mean, dtype=dtype, device=device)
    std = torch.tensor(std, dtype=dtype, device=device)
    return mean, std


############################################################################


def get_state(positions, rotations, foot_contact):
    """
    Args:
        positions (tensor): (batch, seq, joint, 3)
        rotation (tensor): (batch, seq, joint, 3, 3)
        foot_contact ([type]): (batch, seq, 4)
    Returns:
        state (tensor): (batch, seq, dim)
    """
    # (batch, seq, joint, 6)
    rot_6d = data_utils.matrix9D_to_6D_torch(rotations)
    # (batch, seq, joint*6)
    rot = rot_6d.flatten(start_dim=-2)
    state = torch.cat([rot, positions[:, :, 0, :], foot_contact], dim=-1)
    return state


def get_offset(state, target, indices):
    """
    Args:
        state (tensor): (batch, seq, dim) or (batch, dim)
        target (tensor): (batch, 1, dim) or (batch, dim)
        indices (dict): config which defines the meaning of input's dimensions
    """
    offset_slice = slice(indices["r_start_idx"], indices["p_end_idx"])
    offset = target[..., offset_slice] - state[..., offset_slice]
    return offset


def get_transition(model_g, state_context, target, trans_len, indices,
                   tta_emb, noise_emb=None, include_target_frame=False):
    """
    Get motion transition from context frames.
    Args:
        model_g (RmiMotionGenerator): generator model
        state_context (tensor): (batch, context_len, dim),
            states of context frames
        target (tensor): (batch, dim),
        trans_len (int): transition length
        indices (dict): config which defines the meaning of input's dimensions
        tta_emb (TimeToArrivalEmbedding):
            time-to-arrival embedding module
        noise_emb (TargetNoiseEmbedding, optional):
            target noise embedding module. When None, no noise will be added.
            Defaults to None.
        include_target_frame (bool, optional):
            When True, the returned state include prediction at target frame

    Returns:
        tensor: states: (batch, trans_len, dim)
    """
    states = []
    context_len = state_context.shape[-2]

    # warm up the model using context frames
    hidden = None
    for i in range(context_len):
        state = state_context[..., i, :]            # (batch, dim)
        offset = get_offset(state, target, indices)

        tta = context_len + trans_len - i
        if noise_emb:
            ztarget = noise_emb(tta)
        else:
            ztarget = None
        ztta = tta_emb(tta)

        state, hidden = model_g(state, offset, target, hidden, indices,
                                ztta, ztarget)

    for i in range(trans_len):
        # store previously generated state
        states.append(state[..., None, :])

        # new offset
        offset = get_offset(state, target, indices)

        tta = trans_len - i
        if noise_emb:
            ztarget = noise_emb(tta)
        else:
            ztarget = None
        ztta = tta_emb(tta)

        state, hidden = model_g(state, offset, target, hidden, indices,
                                ztta, ztarget)

    if include_target_frame:
        states.append(state[..., None, :])

    return torch.cat(states, dim=-2)


def get_train_stats(config, use_cache=True, stats_folder=None,
                    dataset_name="train_stats"):
    context_len = config["train"]["context_len"]
    if stats_folder is None:
        stats_folder = config["workspace"]
    stats_path = os.path.join(stats_folder, "train_stats_rmi.pkl")

    if use_cache and os.path.exists(stats_path):
        with open(stats_path, "rb") as fh:
            train_stats = pickle.load(fh)
        print("Train stats load from {}".format(stats_path))
    else:
        # calculate training stats
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset, data_loader = train_utils.init_bvh_dataset(
            config, dataset_name, device, shuffle=True, dtype=torch.float64)

        states = []
        c_slice = slice(config["indices"]["c_start_idx"],
                        config["indices"]["c_end_idx"])
        for i, data in enumerate(data_loader, 0):
            (positions, rotations, global_positions, global_rotations,
             foot_contact, parents, data_idx) = data
            positions, rotations = data_utils.to_start_centered_data(
                positions, rotations, context_len)
            state = get_state(positions, rotations, foot_contact)
            states.append(state.cpu().numpy())

        states = np.concatenate(states, axis=0)

        mean = np.mean(states, axis=(0, 1))
        std = np.std(states, axis=(0, 1))
        # no need to calculate mean and std of foot_contact in state
        mean[..., c_slice] = 0
        std[..., c_slice] = 1

        train_stats = {
            "mean": mean,
            "std": std,
        }

        with open(stats_path, "wb") as fh:
            pickle.dump(train_stats, fh)

        print("Train stats wrote to {}".format(stats_path))

    return (
        train_stats["mean"], train_stats["std"],
    )


def get_train_stats_torch(config, dtype, device,
                          use_cache=True, stats_folder=None,
                          dataset_name="train_stats"):
    mean, std = get_train_stats(
        config, use_cache, stats_folder, dataset_name)
    mean = torch.tensor(mean, dtype=dtype, device=device)
    std = torch.tensor(std, dtype=dtype, device=device)

    return mean, std


def train(config):
    indices = config["indices"]
    c_slice = slice(indices["c_start_idx"], indices["c_end_idx"])

    info_interval = config["visdom"]["interval"]
    eval_interval = config["visdom"]["interval_eval"]
    eval_trans = config["visdom"]["eval_trans"]

    context_len = config["train"]["context_len"]
    min_trans = config["train"]["min_trans"]
    max_trans = config["train"]["max_trans"]
    max_trans_n_epoch = config["train"]["max_trans_n_epoch"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dataset
    dataset, data_loader = train_utils.init_bvh_dataset(
        config, "train", device, shuffle=True)
    dtype = dataset.dtype

    val_dataset_names, _, val_data_loaders = \
        train_utils.get_val_datasets(config, device, shuffle=False)

    # visualization
    vis, info_idx = train_utils.init_visdom(config)

    # initialize model
    model_g = RmiMotionGenerator(config["model_g"]).to(device)
    model_ds = RmiMotionDiscriminator(config["model_ds"]).to(device)
    model_dl = RmiMotionDiscriminator(config["model_dl"]).to(device)

    d_encoder_out = config["model_g"]["d_encoder_out"]
    noise_sigma = config["model_g"]["noise_sigma"]
    tta_emb = TimeToArrivalEmbedding(
        d_encoder_out, context_len, max_trans).to(device)

    if config["train"]["target_noise"]:
        noise_emb = TargetNoiseEmbedding(
            d_encoder_out * 2, noise_sigma).to(device)
    else:
        noise_emb = None

    # initialize optimizer
    optimizer_g = Adam(model_g.parameters(),
                       lr=config["train"]["lr"],
                       betas=config["train"]["betas"],
                       amsgrad=config["train"]["amsgrad"])
    optimizer_ds = Adam(model_ds.parameters(),
                        lr=config["train"]["lr"],
                        betas=config["train"]["betas"],
                        amsgrad=config["train"]["amsgrad"])
    optimizer_dl = Adam(model_dl.parameters(),
                        lr=config["train"]["lr"],
                        betas=config["train"]["betas"],
                        amsgrad=config["train"]["amsgrad"])

    # load checkpoint
    epoch, iteration = train_utils.load_checkpoint(
        config, model_g, optimizer_g)
    train_utils.load_checkpoint(config, model_ds, optimizer_ds, suffix=".ds")
    train_utils.load_checkpoint(config, model_dl, optimizer_dl, suffix=".dl")

    # make sure loaded optimizers' learning rate matches what is set in config
    optimizers = [optimizer_g, optimizer_ds, optimizer_dl]
    for optim in optimizers:
        for g in optim.param_groups:
            g['lr'] = config["train"]["lr"]
            g['betas'] = config["train"]["betas"]
            g['amsgrad'] = config["train"]["amsgrad"]

    # train stats
    mean, std = get_train_stats_torch(config, dtype, device)

    l2_loss = torch.nn.MSELoss(reduction="mean")

    loss_avg = 0
    p_loss_avg = 0
    r_loss_avg = 0
    c_loss_avg = 0
    d_loss_avg = 0
    dl_loss_avg = 0
    ds_loss_avg = 0
    model_grad_clipped_avg = 0
    model_grad_raw_avg = 0
    model_grad_max = -1

    min_val_loss = float("inf")

    while epoch < config["train"]["total_epoch"]:
        curr_max_trans = int(
            min_trans + (max_trans - min_trans) * (
                min(1, max(0, (epoch - 1) / max_trans_n_epoch))
            )
        )

        for i, data in enumerate(data_loader, 0):
            (positions, rotations, global_positions, global_rotations,
             foot_contact, parents, data_idx) = data
            parents = parents[0]

            positions, rotations = data_utils.to_start_centered_data(
                positions, rotations, context_len)
            global_rotations, global_positions = data_utils.fk_torch(
                rotations, positions, parents)

            batch_size = positions.shape[0]
            ones = torch.ones(batch_size, 1).to(device)
            zeros = torch.zeros(batch_size, 1).to(device)

            # randomize transition length
            trans_len = random.randint(min_trans, curr_max_trans)
            target_idx = context_len + trans_len
            seq_slice = slice(context_len, context_len + trans_len + 1)

            state_gt = get_state(positions, rotations, foot_contact)

            # apply zscore normalization
            state_gt_zscore = (state_gt - mean) / std
            state_context = state_gt_zscore[..., :context_len, :]
            target = state_gt_zscore[..., target_idx, :]

            # train discriminator ###########################################
            optimizer_ds.zero_grad()
            optimizer_dl.zero_grad()
            model_ds.train()
            model_dl.train()
            model_g.eval()

            with torch.no_grad():
                state_fake = get_transition(
                    model_g, state_context, target, trans_len, indices,
                    tta_emb, noise_emb, include_target_frame=True)
                state_fake = torch.cat([state_context, state_fake], dim=-2)

            state_real = state_gt_zscore[..., :target_idx + 1, :]

            assert state_real.shape == state_fake.shape

            ds_loss = (
                l2_loss(model_ds(state_real), ones) +
                l2_loss(model_ds(state_fake), zeros)
            )

            dl_loss = (
                l2_loss(model_dl(state_real), ones) +
                l2_loss(model_dl(state_fake), zeros)
            )

            ds_loss.backward()
            dl_loss.backward()
            optimizer_ds.step()
            optimizer_dl.step()

            # train generator ###############################################
            optimizer_g.zero_grad()
            model_dl.eval()
            model_ds.eval()
            model_g.train()

            # (batch, context_len+trans_len+1, dim)
            state_out_zscore = get_transition(
                model_g, state_context, target, trans_len, indices,
                tta_emb, noise_emb, include_target_frame=True)
            state_out_zscore = torch.cat(
                [state_context, state_out_zscore], dim=-2)

            # discriminator loss
            d_loss = (
                l2_loss(model_ds(state_out_zscore), ones) +
                l2_loss(model_dl(state_out_zscore), ones)
            ) / 2

            # undo zscore
            state_out = state_out_zscore * std + mean

            pos_new = train_utils.get_new_positions(
                positions, state_out, indices, seq_slice)
            rot_new = train_utils.get_new_rotations(
                state_out, indices, rotations, seq_slice)
            c_new = state_out[..., c_slice]

            grot_new, gpos_new = data_utils.fk_torch(rot_new, pos_new, parents)

            r_loss = train_utils.cal_r_loss(
                state_gt, state_out, seq_slice, indices)
            p_loss = train_utils.cal_p_loss(
                global_positions, gpos_new, seq_slice)
            c_loss = train_utils.cal_c_loss(
                foot_contact, c_new, seq_slice)

            # loss
            loss = (
                config["weights"]["rw"] * r_loss +
                config["weights"]["pw"] * p_loss +
                config["weights"]["cw"] * c_loss +
                config["weights"]["dw"] * d_loss
            )

            loss.backward()

            # clip gradient to prevent gradient explosion
            model_grad_norm = torch.nn.utils.clip_grad_norm_(
                model_g.parameters(), config["train"]["max_grad_norm"])

            optimizer_g.step()

            r_loss_avg += r_loss.item()
            p_loss_avg += p_loss.item()
            c_loss_avg += c_loss.item()
            d_loss_avg += d_loss.item()
            loss_avg += loss.item()
            model_grad_raw_avg += model_grad_norm.item()
            model_grad_clipped_avg += min(model_grad_norm.item(),
                                          config["train"]["max_grad_norm"])
            model_grad_max = max(model_grad_max, model_grad_norm.item())
            ds_loss_avg += ds_loss.item()
            dl_loss_avg += dl_loss.item()

            if iteration % config["train"]["checkpoint_interval"] == 0:
                train_utils.save_checkpoint(
                    config, model_g, epoch, iteration, optimizer_g)
                train_utils.save_checkpoint(
                    config, model_ds, epoch, iteration, optimizer_ds,
                    suffix=".ds")
                train_utils.save_checkpoint(
                    config, model_dl, epoch, iteration,
                    optimizer_dl, suffix=".dl")
                vis.save([config["visdom"]["env"]])

            if iteration % info_interval == 0:
                r_loss_avg /= info_interval
                p_loss_avg /= info_interval
                c_loss_avg /= info_interval
                d_loss_avg /= info_interval
                loss_avg /= info_interval
                model_grad_raw_avg /= info_interval
                model_grad_clipped_avg /= info_interval
                ds_loss_avg /= info_interval
                dl_loss_avg /= info_interval

                lr = optimizer_g.param_groups[0]["lr"]

                print("Epoch: {}, Iteration: {}, lr: {:.8f}, "
                      "loss: {:.6f}, r: {:.6f}, p: {:.6f}, "
                      "c: {:.6f}, d: {:.6f}, ds: {:.6f}, dl: {:.6f}".format(
                          epoch, iteration, lr, loss_avg,
                          r_loss_avg, p_loss_avg, c_loss_avg,
                          d_loss_avg, ds_loss_avg, dl_loss_avg))

                contents = [
                    ["loss", "r_loss", r_loss_avg],
                    ["loss", "p_loss", p_loss_avg],
                    ["loss", "c_loss", c_loss_avg],
                    ["loss", "d_loss", d_loss_avg],
                    ["loss discriminator", "ds_loss", ds_loss_avg],
                    ["loss discriminator", "dl_loss", dl_loss_avg],
                    ["loss weighted", "r_loss",
                        r_loss_avg * config["weights"]["rw"]],
                    ["loss weighted", "p_loss",
                        p_loss_avg * config["weights"]["pw"]],
                    ["loss weighted", "c_loss",
                        c_loss_avg * config["weights"]["cw"]],
                    ["loss weighted", "d_loss",
                        d_loss_avg * config["weights"]["dw"]],
                    ["loss weighted", "loss", loss_avg],
                    ["learning rate", "lr", lr],
                    ["epoch", "epoch", epoch],
                    ["iterations", "iterations", iteration],
                    ["transition", "length", trans_len],
                    ["transition", "curr_max_trans", curr_max_trans],
                    ["gradient", "raw_avg", model_grad_raw_avg],
                    ["gradient", "clipped_avg", model_grad_clipped_avg],
                    ["gradient", "max", model_grad_max],
                ]

                if iteration % eval_interval == 0:
                    for trans in eval_trans:
                        print("\ntrans: {}".format(trans))

                        for i in range(len(val_data_loaders)):
                            ds_name = val_dataset_names[i]
                            ds_loader = val_data_loaders[i]
                            gpos_loss, gquat_loss, npss_loss = eval_on_dataset(
                                config, ds_loader, model_g, trans)

                            contents.extend([
                                [ds_name, "gpos_{}".format(trans),
                                 gpos_loss],
                                [ds_name, "gquat_{}".format(trans),
                                 gquat_loss],
                                [ds_name, "npss_{}".format(trans),
                                 npss_loss],
                            ])
                            print("{}:\ngpos: {:6f}, gquat: {:6f}, "
                                  "npss: {:.6f}".format(ds_name, gpos_loss,
                                                        gquat_loss, npss_loss))

                            if ds_name == "val":
                                # After iterations, val_loss will be the
                                # sum of losses on dataset named "val"
                                # with transition length equals to last value
                                # in eval_interval.
                                val_loss = (gpos_loss + gquat_loss +
                                            npss_loss)

                    if min_val_loss > val_loss:
                        min_val_loss = val_loss
                        # save min loss checkpoint
                        train_utils.save_checkpoint(
                            config, model_g, epoch, iteration, optimizer_g,
                            suffix=".min", n_ckp=3)
                        train_utils.save_checkpoint(
                            config, model_ds, epoch, iteration, optimizer_ds,
                            suffix=".ds.min", n_ckp=3)
                        train_utils.save_checkpoint(
                            config, model_dl, epoch, iteration,
                            optimizer_dl, suffix=".dl.min", n_ckp=3)

                train_utils.to_visdom(vis, info_idx, contents)
                r_loss_avg = 0
                p_loss_avg = 0
                c_loss_avg = 0
                d_loss_avg = 0
                loss_avg = 0
                model_grad_raw_avg = 0
                model_grad_clipped_avg = 0
                model_grad_max = -1
                ds_loss_avg = 0
                dl_loss_avg = 0

                info_idx += 1

            iteration += 1

        epoch += 1


def eval_on_dataset(config, data_loader, model_g,
                    trans_len, debug=False):
    device = data_loader.dataset.device
    dtype = data_loader.dataset.dtype

    indices = config["indices"]
    max_trans = config["train"]["max_trans"]
    context_len = config["train"]["context_len"]
    d_encoder_out = config["model_g"]["d_encoder_out"]
    target_idx = context_len + trans_len
    seq_slice = slice(context_len, target_idx)

    tta_emb = TimeToArrivalEmbedding(
        d_encoder_out, context_len, max_trans).to(device)

    # train stats
    mean_state, std_state = get_train_stats_torch(config, dtype, device)
    mean_rmi, std_rmi = get_rmi_benchmark_stats_torch(
        config, dtype, device)

    data_indexes = []
    gpos_loss = []
    gquat_loss = []
    npss_loss = []
    npss_weights = []

    for i, data in enumerate(data_loader, 0):
        (positions, rotations, global_positions, global_rotations,
            foot_contact, parents, data_idx) = data
        parents = parents[0]

        positions, rotations = data_utils.to_start_centered_data(
            positions, rotations, context_len)

        pos_new, rot_new, c_new = evaluate(
            model_g, positions, rotations, foot_contact, seq_slice,
            indices, mean_state, std_state, tta_emb)

        (gpos_batch_loss, gquat_batch_loss,
         npss_batch_loss, npss_batch_weights) = \
            benchmark.get_rmi_style_batch_loss(
                positions, rotations, pos_new, rot_new, parents,
                context_len, target_idx, mean_rmi, std_rmi
        )

        gpos_loss.append(gpos_batch_loss)
        gquat_loss.append(gquat_batch_loss)
        npss_loss.append(npss_batch_loss)
        npss_weights.append(npss_batch_weights)
        data_indexes.extend(data_idx.tolist())

    gpos_loss = np.concatenate(gpos_loss, axis=-1)
    gquat_loss = np.concatenate(gquat_loss, axis=-1)

    npss_loss = np.concatenate(npss_loss, axis=0)           # (batch, dim)
    npss_weights = np.concatenate(npss_weights, axis=0)
    npss_weights = npss_weights / np.sum(npss_weights)      # (batch, dim)
    npss_loss = np.sum(npss_loss * npss_weights, axis=-1)   # (batch, )

    if debug:
        total_loss = gpos_loss + gquat_loss + npss_loss
        loss_data = list(zip(
            total_loss.tolist(),
            gpos_loss.tolist(),
            gquat_loss.tolist(),
            npss_loss.tolist(),
            data_indexes
        ))
        loss_data.sort()
        loss_data.reverse()

        return gpos_loss.mean(), gquat_loss.mean(), npss_loss.sum(), loss_data
    else:
        return gpos_loss.mean(), gquat_loss.mean(), npss_loss.sum()


def evaluate(model_g, positions, rotations, foot_contact, seq_slice, indices,
             mean, std, tta_emb, noise_emb=None):
    """
    Generate transition animation using Robust Motion Inbetweening's method
    (Harvey et. al., 2020).

    positions and rotation should already been preprocessed using
    motion_inbetween.data.utils.to_start_centered_data().

    positions[..., seq_slice, :, :] value are useless data. It is meant to
    be generated by this function. When inputing into this function, it can be
    actual ground truth data fetched from a dataset, even though it won't
    take any effect. Or it can be set to zero as placeholder values.

    The same applies to rotations[..., seq_slice, :, :, :] and
    foot_contact[..., seq_slice, :]. Specially for rotations, when setting
    placeholder values, make sure last two dimesions are set to identity
    matrices.

    Args:
        model_g (nn.Module): generator model
        positions (tensor): (batch, seq, joint, 3)
        rotations (tensor): (batch, seq, joint, 3, 3)
        foot_contact (tensor): (batch, seq, 4)
        seq_slice (slice): sequence slice where motion will be predicted
        indices (dict): config which defines the meaning of input's dimensions
        mean (tensor): mean of model input
        std (tensor): std of model input
        tta_emb (TimeToArrivalEmbedding):
            time-to-arrival embedding module
        noise_emb (TargetNoiseEmbedding, optional):
            target noise embedding module. When None, no noise will be added.
            Defaults to None.

    Returns:
        tensor, tensor, tensor: new positions, new rotations and new
        foot_contact with predicted animation. Shape same as input.
    """
    context_len = seq_slice.start
    target_idx = seq_slice.stop
    trans_len = target_idx - context_len
    c_slice = slice(indices["c_start_idx"], indices["c_end_idx"])

    with torch.no_grad():
        model_g.eval()

        state_gt = get_state(positions, rotations, foot_contact)

        # apply zscore normalization
        state_gt_zscore = (state_gt - mean) / std
        state_context = state_gt_zscore[..., :context_len, :]
        target = state_gt_zscore[..., target_idx, :]

        # (batch, context_len+trans_len+1, dim)
        state_out_zscore = get_transition(
            model_g, state_context, target, trans_len, indices,
            tta_emb, noise_emb, include_target_frame=False)
        state_out_zscore = torch.cat(
            [state_context, state_out_zscore], dim=-2)

        # undo zscore
        state_out = state_out_zscore * std + mean

        pos_new = train_utils.get_new_positions(
            positions, state_out, indices, seq_slice)
        rot_new = train_utils.get_new_rotations(
            state_out, indices, rotations, seq_slice)
        c_new = foot_contact.clone().detach()
        c_new[..., seq_slice, :] = state_out[..., seq_slice, c_slice]

        return pos_new, rot_new, c_new
