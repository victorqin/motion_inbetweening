import os
import pickle
import random
import numpy as np

import torch
from torch.optim import Adam

from motion_inbetween import benchmark
from motion_inbetween.model import ContextTransformer
from motion_inbetween.data import utils_torch as data_utils
from motion_inbetween.train import rmi
from motion_inbetween.train import utils as train_utils


def get_model_input(positions, rotations):
    # positions: (batch, seq, joint, 3)
    # rotation: (batch, seq, joint, 3, 3)
    # return (batch, seq, joint*6+3)
    rot_6d = data_utils.matrix9D_to_6D_torch(rotations)
    rot = rot_6d.flatten(start_dim=-2)
    x = torch.cat([rot, positions[:, :, 0, :]], dim=-1)
    return x


def get_train_stats(config, use_cache=True, stats_folder=None,
                    dataset_name="train_stats"):
    context_len = config["train"]["context_len"]
    if stats_folder is None:
        stats_folder = config["workspace"]
    stats_path = os.path.join(stats_folder, "train_stats_context.pkl")

    if use_cache and os.path.exists(stats_path):
        with open(stats_path, "rb") as fh:
            train_stats = pickle.load(fh)
        print("Train stats load from {}".format(stats_path))
    else:
        # calculate training stats
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset, data_loader = train_utils.init_bvh_dataset(
            config, dataset_name, device, shuffle=True, dtype=torch.float64)

        input_data = []
        for i, data in enumerate(data_loader, 0):
            (positions, rotations, global_positions, global_rotations,
             foot_contact, parents, data_idx) = data
            parents = parents[0]

            positions, rotations = data_utils.to_start_centered_data(
                positions, rotations, context_len)
            x = get_model_input(positions, rotations)
            input_data.append(x.cpu().numpy())

        input_data = np.concatenate(input_data, axis=0)

        mean = np.mean(input_data, axis=(0, 1))
        std = np.std(input_data, axis=(0, 1))

        train_stats = {
            "mean": mean,
            "std": std
        }

        with open(stats_path, "wb") as fh:
            pickle.dump(train_stats, fh)

        print("Train stats wrote to {}".format(stats_path))

    return train_stats["mean"], train_stats["std"]


def get_train_stats_torch(config, dtype, device,
                          use_cache=True, stats_folder=None,
                          dataset_name="train_stats"):
    mean, std = get_train_stats(config, use_cache, stats_folder, dataset_name)
    mean = torch.tensor(mean, dtype=dtype, device=device)
    std = torch.tensor(std, dtype=dtype, device=device)
    return mean, std


def get_midway_targets(seq_slice, midway_targets_amount, midway_targets_p):
    targets = set()
    trans_len = seq_slice.stop - seq_slice.start
    midway_targets_amount = int(midway_targets_amount * trans_len)
    for i in range(midway_targets_amount):
        if random.random() < midway_targets_p:
            targets.add(random.randrange(seq_slice.start, seq_slice.stop))
    return list(targets)


def get_attention_mask(window_len, context_len, target_idx, device,
                       midway_targets=()):
    atten_mask = torch.ones(window_len, window_len,
                            device=device, dtype=torch.bool)
    atten_mask[:, target_idx] = False
    atten_mask[:, :context_len] = False
    atten_mask[:, midway_targets] = False
    atten_mask = atten_mask.unsqueeze(0)

    # (1, seq, seq)
    return atten_mask


def get_data_mask(window_len, d_mask, constrained_slices,
                  context_len, target_idx, device, dtype,
                  midway_targets=()):
    # 0 for unknown and 1 for known
    data_mask = torch.zeros((window_len, d_mask), device=device, dtype=dtype)
    data_mask[:context_len, :] = 1
    data_mask[target_idx, :] = 1

    for s in constrained_slices:
        data_mask[midway_targets, s] = 1

    # (seq, d_mask)
    return data_mask


def get_keyframe_pos_indices(window_len, seq_slice, dtype, device):
    # position index relative to context and target frame
    ctx_idx = torch.arange(window_len, dtype=dtype, device=device)
    ctx_idx = ctx_idx - (seq_slice.start - 1)
    ctx_idx = ctx_idx[..., None]

    tgt_idx = torch.arange(window_len, dtype=dtype, device=device)
    tgt_idx = -(tgt_idx - seq_slice.stop)
    tgt_idx = tgt_idx[..., None]

    # ctx_idx: (seq, 1), tgt_idx: (seq, 1)
    keyframe_pos_indices = torch.cat([ctx_idx, tgt_idx], dim=-1)

    # (1, seq, 2)
    return keyframe_pos_indices[None]


def set_placeholder_root_pos(x, seq_slice, midway_targets, p_slice):
    # set root position of missing part to linear interpolation of
    # root position between constrained frames (i.e. last context frame,
    # midway target frames and target frame).
    constrained_frames = [seq_slice.start - 1, seq_slice.stop]
    constrained_frames.extend(midway_targets)
    constrained_frames.sort()
    for i in range(len(constrained_frames) - 1):
        start_idx = constrained_frames[i]
        end_idx = constrained_frames[i + 1]
        start_slice = slice(start_idx, start_idx + 1)
        end_slice = slice(end_idx, end_idx + 1)
        inbetween_slice = slice(start_idx + 1, end_idx)

        x[..., inbetween_slice, p_slice] = \
            benchmark.get_linear_interpolation(
                x[..., start_slice, p_slice],
                x[..., end_slice, p_slice],
                end_idx - start_idx - 1
        )
    return x


def train(config):
    indices = config["indices"]
    info_interval = config["visdom"]["interval"]
    eval_interval = config["visdom"]["interval_eval"]
    eval_trans = config["visdom"]["eval_trans"]
    p_slice = slice(indices["p_start_idx"], indices["p_end_idx"])

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
    model = ContextTransformer(config["model"]).to(device)

    # initialize optimizer
    optimizer = Adam(model.parameters(), lr=config["train"]["lr"])

    # learning rate scheduler
    scheduler = train_utils.get_noam_lr_scheduler(config, optimizer)

    # load checkpoint
    epoch, iteration = train_utils.load_checkpoint(
        config, model, optimizer, scheduler)

    # training stats
    mean, std = get_train_stats_torch(config, dtype, device)

    window_len = config["datasets"]["train"]["window"]
    context_len = config["train"]["context_len"]
    min_trans = config["train"]["min_trans"]
    max_trans = config["train"]["max_trans"]
    midway_targets_amount = config["train"]["midway_targets_amount"]
    midway_targets_p = config["train"]["midway_targets_p"]

    loss_avg = 0
    p_loss_avg = 0
    r_loss_avg = 0
    smooth_loss_avg = 0

    min_val_loss = float("inf")

    while epoch < config["train"]["total_epoch"]:
        for i, data in enumerate(data_loader, 0):
            (positions, rotations, global_positions, global_rotations,
             foot_contact, parents, data_idx) = data
            parents = parents[0]

            positions, rotations = data_utils.to_start_centered_data(
                positions, rotations, context_len)
            global_rotations, global_positions = data_utils.fk_torch(
                rotations, positions, parents)

            # randomize transition length
            trans_len = random.randint(min_trans, max_trans)
            target_idx = context_len + trans_len
            seq_slice = slice(context_len, target_idx)

            # get random midway target frames
            midway_targets = get_midway_targets(
                seq_slice, midway_targets_amount, midway_targets_p)

            # attention mask
            atten_mask = get_attention_mask(
                window_len, context_len, target_idx, device,
                midway_targets=midway_targets)

            # data mask
            data_mask = get_data_mask(
                window_len, model.d_mask, model.constrained_slices,
                context_len, target_idx, device, dtype, midway_targets)

            # position index relative to context and target frame
            keyframe_pos_idx = get_keyframe_pos_indices(
                window_len, seq_slice, dtype, device)

            # prepare model input
            x_gt = get_model_input(positions, rotations)
            x_gt_zscore = (x_gt - mean) / std

            x = torch.cat([
                x_gt_zscore * data_mask,
                data_mask.expand(*x_gt_zscore.shape[:-1], data_mask.shape[-1])
            ], dim=-1)

            x = set_placeholder_root_pos(x, seq_slice, midway_targets, p_slice)

            # calculate model output y
            optimizer.zero_grad()
            model.train()

            model_out = model(x, keyframe_pos_idx, mask=atten_mask)
            y = x_gt_zscore.clone().detach()
            y[..., seq_slice, :] = model_out[..., seq_slice, :]

            y = y * std + mean

            pos_new = train_utils.get_new_positions(positions, y, indices)
            rot_new = train_utils.get_new_rotations(y, indices)

            grot_new, gpos_new = data_utils.fk_torch(rot_new, pos_new, parents)

            r_loss = train_utils.cal_r_loss(x_gt, y, seq_slice, indices)
            smooth_loss = train_utils.cal_smooth_loss(gpos_new, seq_slice)
            p_loss = train_utils.cal_p_loss(
                global_positions, gpos_new, seq_slice)

            # loss
            loss = (
                config["weights"]["rw"] * r_loss +
                config["weights"]["pw"] * p_loss +
                config["weights"]["sw"] * smooth_loss
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            r_loss_avg += r_loss.item()
            p_loss_avg += p_loss.item()
            smooth_loss_avg += smooth_loss.item()
            loss_avg += loss.item()

            if iteration % config["train"]["checkpoint_interval"] == 0:
                train_utils.save_checkpoint(config, model, epoch, iteration,
                                            optimizer, scheduler)
                vis.save([config["visdom"]["env"]])

            if iteration % info_interval == 0:
                r_loss_avg /= info_interval
                p_loss_avg /= info_interval
                smooth_loss_avg /= info_interval
                loss_avg /= info_interval
                lr = optimizer.param_groups[0]["lr"]

                print("Epoch: {}, Iteration: {}, lr: {:.8f}, "
                      "loss: {:.6f}, r: {:.6f}, p: {:.6f}, "
                      "smooth: {:.6f}".format(
                          epoch, iteration, lr, loss_avg,
                          r_loss_avg, p_loss_avg, smooth_loss_avg))

                contents = [
                    ["loss", "r_loss", r_loss_avg],
                    ["loss", "p_loss", p_loss_avg],
                    ["loss", "smooth_loss", smooth_loss_avg],
                    ["loss weighted", "r_loss",
                        r_loss_avg * config["weights"]["rw"]],
                    ["loss weighted", "p_loss",
                        p_loss_avg * config["weights"]["pw"]],
                    ["loss weighted", "smooth_loss",
                        smooth_loss_avg * config["weights"]["sw"]],
                    ["loss weighted", "loss", loss_avg],
                    ["learning rate", "lr", lr],
                    ["epoch", "epoch", epoch],
                    ["iterations", "iterations", iteration],
                ]

                if iteration % eval_interval == 0:
                    for trans in eval_trans:
                        print("trans: {}\n".format(trans))

                        for i in range(len(val_data_loaders)):
                            ds_name = val_dataset_names[i]
                            ds_loader = val_data_loaders[i]

                            gpos_loss, gquat_loss, npss_loss = eval_on_dataset(
                                config, ds_loader, model, trans)

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
                            config, model, epoch, iteration,
                            optimizer, scheduler, suffix=".min", n_ckp=3)

                train_utils.to_visdom(vis, info_idx, contents)
                r_loss_avg = 0
                p_loss_avg = 0
                smooth_loss_avg = 0
                loss_avg = 0
                info_idx += 1

            iteration += 1

        epoch += 1


def eval_on_dataset(config, data_loader, model, trans_len,
                    debug=False, post_process=True):
    device = data_loader.dataset.device
    dtype = data_loader.dataset.dtype

    indices = config["indices"]
    context_len = config["train"]["context_len"]
    target_idx = context_len + trans_len
    seq_slice = slice(context_len, target_idx)
    window_len = context_len + trans_len + 2

    mean, std = get_train_stats_torch(config, dtype, device)
    mean_rmi, std_rmi = rmi.get_rmi_benchmark_stats_torch(
        config, dtype, device)

    # attention mask
    atten_mask = get_attention_mask(
        window_len, context_len, target_idx, device)

    data_indexes = []
    gpos_loss = []
    gquat_loss = []
    npss_loss = []
    npss_weights = []

    for i, data in enumerate(data_loader, 0):
        (positions, rotations, global_positions, global_rotations,
            foot_contact, parents, data_idx) = data
        parents = parents[0]

        positions = positions[..., :window_len, :, :]
        rotations = rotations[..., :window_len, :, :, :]
        global_positions = global_positions[..., :window_len, :, :]
        global_rotations = global_rotations[..., :window_len, :, :, :]
        foot_contact = foot_contact[..., :window_len, :]

        positions, rotations = data_utils.to_start_centered_data(
            positions, rotations, context_len)

        pos_new, rot_new = evaluate(
            model, positions, rotations, seq_slice,
            indices, mean, std, atten_mask, post_process)

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

    gpos_loss = np.concatenate(gpos_loss, axis=0)
    gquat_loss = np.concatenate(gquat_loss, axis=0)

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


def evaluate(model, positions, rotations, seq_slice, indices,
             mean, std, atten_mask, post_process=True,
             midway_targets=()):
    """
    Generate transition animation.

    positions and rotation should already been preprocessed using
    motion_inbetween.data.utils.to_start_centered_data().

    positions, shape: (batch, seq, joint, 3) and
    rotations, shape: (batch, seq, joint, 3, 3)
    is a mixture of ground truth and placeholder data.

    There are two constraint mode:
    1) fully constrained: at constrained midway target frames, input values of
        all joints should be the same as output values.
    2) partially constrained: only selected joints' input value are kept in
        output (e.g. only constrain root joint).

    At context frames, target frame and midway target frames, ground truth
    data should be provided (in partially constrained mode, only provide ground
    truth value of constrained dimensions).

    The placeholder values:
    - for positions: set to zero
    - for rotations: set to identity matrix

    Args:
        model (nn.Module): context model
        positions (tensor): (batch, seq, joint, 3)
        rotations (tensor): (batch, seq, joint, 3, 3)
        seq_slice (slice): sequence slice where motion will be predicted
        indices (dict): config which defines the meaning of input's dimensions
        mean (tensor): mean of model input
        std (tensor): std of model input
        atten_mask (tensor): (1, seq, seq) model attention mask
        post_process (bool): Whether post processing is enabled or not.
            Defaults to True.
        midway_targets (list of int): list of midway targets (constrained
            frames indexes).

    Returns:
        tensor, tensor: new positions, new rotations with predicted animation
        with same shape as input.
    """
    dtype = positions.dtype
    device = positions.device
    window_len = positions.shape[-3]
    context_len = seq_slice.start
    target_idx = seq_slice.stop

    # If current context model is not trained with constrants,
    # ignore midway_targets.
    if midway_targets and not model.constrained_slices:
        print(
            "WARNING: Context model is not trained with constraints, but "
            "midway_targets is provided with values while calling evaluate()! "
            "midway_targets is ignored!"
        )
        midway_targets = []

    with torch.no_grad():
        model.eval()

        if midway_targets:
            midway_targets.sort()
            atten_mask = atten_mask.clone().detach()
            atten_mask[0, :, midway_targets] = False

        # prepare model input
        x_orig = get_model_input(positions, rotations)

        # zscore
        x_zscore = (x_orig - mean) / std

        # data mask (seq, 1)
        data_mask = get_data_mask(
            window_len, model.d_mask, model.constrained_slices, context_len,
            target_idx, device, dtype, midway_targets)

        keyframe_pos_idx = get_keyframe_pos_indices(
            window_len, seq_slice, dtype, device)

        x = torch.cat([
            x_zscore * data_mask,
            data_mask.expand(*x_zscore.shape[:-1], data_mask.shape[-1])
        ], dim=-1)

        p_slice = slice(indices["p_start_idx"], indices["p_end_idx"])
        x = set_placeholder_root_pos(x, seq_slice, midway_targets, p_slice)

        # calculate model output y
        model_out = model(x, keyframe_pos_idx, mask=atten_mask)
        y = x_zscore.clone().detach()
        y[..., seq_slice, :] = model_out[..., seq_slice, :]

        if post_process:
            y = train_utils.anim_post_process(y, x_zscore, seq_slice)

        # reverse zscore
        y = y * std + mean

        # new pos and rot
        pos_new = train_utils.get_new_positions(
            positions, y, indices, seq_slice)
        rot_new = train_utils.get_new_rotations(
            y, indices, rotations, seq_slice)

        return pos_new, rot_new
