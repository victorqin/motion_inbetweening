import os
import math
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from motion_inbetween.model import ContextTransformer, DetailTransformer
from motion_inbetween.data import utils_torch as data_utils
from motion_inbetween.train import rmi
from motion_inbetween.train import utils as train_utils
from motion_inbetween.train import context_model as ctx_mdl
from motion_inbetween.benchmark import get_rmi_style_batch_loss


def get_model_input(positions, rotations):
    """
    Get detail model input. Including state, delta, offset.

    Args:
        positions (tensor): (batch, seq, joint, 3)
        rotations (tensor): (batch, seq, joint, 3, 3)

    Returns:
        tuple: state, delta
            state: (batch, seq, joint*3+3).
                Describe joint rotation and root position.
            delta: (batch, seq, joint*3+3), shape same as state.
                Delta between current frame's state and previous frame's state.
    """
    # state
    rot_6d = data_utils.matrix9D_to_6D_torch(rotations)
    rot = rot_6d.flatten(start_dim=-2)
    state = torch.cat([rot, positions[:, :, 0, :]], dim=-1)

    zeros_shape = list(state.shape)
    zeros_shape[-2] = 1
    zeros = torch.zeros(*zeros_shape, dtype=state.dtype, device=state.device)

    # delta
    delta = state[..., 1:, :] - state[..., :-1, :]
    # Pad zero on the first frame for shape consistency
    delta = torch.cat([zeros, delta], dim=-2)

    return state, delta


def get_model_output(detail_model, state_zscore, data_mask, atten_mask,
                     foot_contact, seq_slice, c_slice, rp_slice):
    data_mask = data_mask.expand(*state_zscore.shape[:-1], data_mask.shape[-1])
    model_out = detail_model(
        torch.cat([state_zscore, data_mask], dim=-1), mask=atten_mask)

    c_out = foot_contact.clone().detach()
    c_out[..., seq_slice, :] = torch.sigmoid(
        model_out[..., seq_slice, c_slice])

    state_out = state_zscore.clone().detach()
    state_out[..., seq_slice, :] = model_out[..., seq_slice, rp_slice]

    return state_out, c_out


def get_discriminator_input(state, foot_contact, p_slice):
    # change root position to root velocity
    root_vel = state[..., 1:, p_slice] - state[..., :-1, p_slice]

    # Pad zero on the first frame for shape consistency
    zero_shape = list(root_vel.shape)
    zero_shape[-2] = 1
    zeros = torch.zeros(*zero_shape, dtype=state.dtype, device=state.device)
    root_vel = torch.cat([zeros, root_vel], dim=-2)

    dis_state = state.clone()
    dis_state[..., :, p_slice] = root_vel

    dis_input = dis_state
    return dis_input


def get_train_stats(config, use_cache=True, stats_folder=None,
                    dataset_name="train_stats"):
    context_len = config["train"]["context_len"]
    if stats_folder is None:
        stats_folder = config["workspace"]
    stats_path = os.path.join(stats_folder, "train_stats_detail.pkl")

    if use_cache and os.path.exists(stats_path):
        with open(stats_path, "rb") as fh:
            train_stats = pickle.load(fh)
        print("Train stats load from {}".format(stats_path))
    else:
        # calculate training stats of state, delta
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset, data_loader = train_utils.init_bvh_dataset(
            config, dataset_name, device, shuffle=True, dtype=torch.float64)

        state_data = []
        delta_data = []

        for i, data in enumerate(data_loader, 0):
            (positions, rotations, global_positions, global_rotations,
             foot_contact, parents, data_idx) = data
            parents = parents[0]

            positions, rotations = data_utils.to_start_centered_data(
                positions, rotations, context_len)

            state, delta = get_model_input(positions, rotations)
            state_data.append(state.cpu().numpy())
            delta_data.append(delta.cpu().numpy())

        state_data = np.concatenate(state_data, axis=0)
        delta_data = np.concatenate(delta_data, axis=0)

        train_stats = {
            "state": {
                "mean": np.mean(state_data, axis=(0, 1)),
                "std": np.std(state_data, axis=(0, 1)),
            },
            "delta": {
                "mean": np.mean(delta_data, axis=(0, 1)),
                "std": np.std(delta_data, axis=(0, 1)),
            }
        }

        with open(stats_path, "wb") as fh:
            pickle.dump(train_stats, fh)

        print("Train stats wrote to {}".format(stats_path))

    return (
        train_stats["state"]["mean"],
        train_stats["state"]["std"],
        train_stats["delta"]["mean"],
        train_stats["delta"]["std"],
    )


def get_train_stats_torch(config, dtype, device,
                          use_cache=True, stats_folder=None,
                          dataset_name="train_stats"):
    mean_state, std_state, mean_delta, std_delta = get_train_stats(
        config, use_cache, stats_folder, dataset_name)

    mean_state = torch.tensor(mean_state, dtype=dtype, device=device)
    std_state = torch.tensor(std_state, dtype=dtype, device=device)
    mean_delta = torch.tensor(mean_delta, dtype=dtype, device=device)
    std_delta = torch.tensor(std_delta, dtype=dtype, device=device)

    return mean_state, std_state, mean_delta, std_delta


def get_attention_mask(window_len, target_idx, device):
    atten_mask = torch.ones(window_len, window_len,
                            device=device, dtype=torch.bool)
    atten_mask[:, :target_idx + 1] = False
    atten_mask = atten_mask.unsqueeze(0)

    # (1, seq, seq)
    return atten_mask


def reset_constrained_values(state, state_gt, delta, delta_gt,
                             midway_targets, constrained_slices):
    for s in constrained_slices:
        state[..., midway_targets, s] = state_gt[..., midway_targets, s]
        delta[..., midway_targets, s] = delta_gt[..., midway_targets, s]


def update_dropout_p(model, iteration, config):
    init_p = config["model"]["dropout"]
    max_iterations = config["model"]["dropout_iterations"]
    factor = 1 - max(max_iterations - iteration, 0) / max_iterations
    p = init_p * (1 + math.cos(factor * math.pi)) / 2

    model.dropout = p
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = p


def train(config, context_config):
    indices = config["indices"]
    info_interval = config["visdom"]["interval"]
    eval_interval = config["visdom"]["interval_eval"]
    eval_trans = config["visdom"]["eval_trans"]

    rp_slice = slice(indices["r_start_idx"], indices["p_end_idx"])
    p_slice = slice(indices["p_start_idx"], indices["p_end_idx"])
    c_slice = slice(indices["c_start_idx"], indices["c_end_idx"])

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
    detail_model = DetailTransformer(config["model"]).to(device)
    context_model = ContextTransformer(context_config["model"]).to(device)

    # initialize optimizer
    optimizer = Adam(detail_model.parameters(), lr=config["train"]["lr"])

    # learning rate scheduler
    scheduler = train_utils.get_noam_lr_scheduler(config, optimizer)

    # load checkpoint
    epoch, iteration = train_utils.load_checkpoint(
        config, detail_model, optimizer, scheduler)
    train_utils.load_checkpoint(context_config, context_model)

    # context model training stats
    mean_ctx, std_ctx = ctx_mdl.get_train_stats_torch(
        context_config, dtype, device)

    # detail model training stats
    mean_state, std_state, _, _ = get_train_stats_torch(
        config, dtype, device)

    window_len = config["datasets"]["train"]["window"]
    context_len = config["train"]["context_len"]
    min_trans = config["train"]["min_trans"]
    max_trans = config["train"]["max_trans"]
    midway_targets_amount = config["train"]["midway_targets_amount"]
    midway_targets_p = config["train"]["midway_targets_p"]

    loss_avg = 0
    p_loss_avg = 0
    r_loss_avg = 0
    c_loss_avg = 0
    f_loss_avg = 0

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
            midway_targets = ctx_mdl.get_midway_targets(
                seq_slice, midway_targets_amount, midway_targets_p)

            # attention mask for context model
            atten_mask_ctx = ctx_mdl.get_attention_mask(
                window_len, context_len, target_idx, device,
                midway_targets=midway_targets)

            # attention mask for detail model
            atten_mask = get_attention_mask(window_len, target_idx, device)

            data_mask = ctx_mdl.get_data_mask(
                window_len, detail_model.d_mask,
                detail_model.constrained_slices, context_len, target_idx,
                device, dtype, midway_targets=midway_targets)

            # get context model output
            pos_ctx, rot_ctx = ctx_mdl.evaluate(
                context_model, positions, rotations, seq_slice,
                indices, mean_ctx, std_ctx, atten_mask_ctx,
                post_process=True, midway_targets=midway_targets)

            # detail model inputs
            state_gt, delta_gt = get_model_input(positions, rotations)
            state, delta = get_model_input(pos_ctx, rot_ctx)

            reset_constrained_values(
                state, state_gt, delta, delta_gt,
                midway_targets, detail_model.constrained_slices)

            state_zscore = (state - mean_state) / std_state

            # train details model --------------------------------
            optimizer.zero_grad()
            detail_model.train()

            # update model dropout p
            update_dropout_p(detail_model, iteration, config)

            state_out, c_out = get_model_output(
                detail_model, state_zscore, data_mask, atten_mask,
                foot_contact, seq_slice, c_slice, rp_slice)

            state_out = state_out * std_state + mean_state

            pos_new = train_utils.get_new_positions(
                positions, state_out, indices)
            rot_new = train_utils.get_new_rotations(state_out, indices)

            grot_new, gpos_new = data_utils.fk_torch(rot_new, pos_new, parents)

            r_loss = train_utils.cal_r_loss(
                state_gt, state_out, seq_slice, indices)
            p_loss = train_utils.cal_p_loss(
                global_positions, gpos_new, seq_slice)
            c_loss = train_utils.cal_c_loss(
                foot_contact, c_out, seq_slice)
            f_loss = train_utils.cal_f_loss(gpos_new, c_out, seq_slice)

            # loss
            loss = (
                config["weights"]["rw"] * r_loss +
                config["weights"]["pw"] * p_loss +
                config["weights"]["cw"] * c_loss +
                config["weights"]["fw"] * f_loss
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            r_loss_avg += r_loss.item()
            p_loss_avg += p_loss.item()
            c_loss_avg += c_loss.item()
            f_loss_avg += f_loss.item()
            loss_avg += loss.item()

            if iteration % config["train"]["checkpoint_interval"] == 0:
                train_utils.save_checkpoint(config, detail_model, epoch,
                                            iteration, optimizer, scheduler)
                vis.save([config["visdom"]["env"]])

            if iteration % info_interval == 0:
                r_loss_avg /= info_interval
                p_loss_avg /= info_interval
                c_loss_avg /= info_interval
                f_loss_avg /= info_interval
                loss_avg /= info_interval
                lr = optimizer.param_groups[0]["lr"]

                print("Epoch: {}, Iteration: {}, lr: {:.8f}, dropout: {:.6f}, "
                      "loss: {:.6f}, r: {:.6f}, p: {:.6f}, c: {:.6f}, "
                      "f: {:.6f}".format(
                          epoch, iteration, lr, detail_model.dropout, loss_avg,
                          r_loss_avg, p_loss_avg, c_loss_avg, f_loss_avg))

                contents = [
                    ["loss", "r_loss", r_loss_avg],
                    ["loss", "p_loss", p_loss_avg],
                    ["loss", "c_loss", c_loss_avg],
                    ["loss", "f_loss", f_loss_avg],
                    ["dropout", "p", detail_model.dropout],
                    ["loss weighted", "r_loss",
                        r_loss_avg * config["weights"]["rw"]],
                    ["loss weighted", "p_loss",
                        p_loss_avg * config["weights"]["pw"]],
                    ["loss weighted", "c_loss",
                        c_loss_avg * config["weights"]["cw"]],
                    ["loss weighted", "f_loss",
                        f_loss_avg * config["weights"]["fw"]],
                    ["loss weighted", "loss", loss_avg],
                    ["learning rate", "lr", lr],
                    ["epoch", "epoch", epoch],
                    ["iterations", "iterations", iteration],
                ]

                if iteration % eval_interval == 0:
                    for trans in eval_trans:
                        print("trans: {}".format(trans))

                        for i in range(len(val_data_loaders)):
                            ds_name = val_dataset_names[i]
                            ds_loader = val_data_loaders[i]
                            gpos_loss, gquat_loss, npss_loss = eval_on_dataset(
                                config, ds_loader, detail_model,
                                context_model, trans, post_process=False)
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
                            config, detail_model, epoch, iteration,
                            optimizer, scheduler, suffix=".min", n_ckp=3)

                train_utils.to_visdom(vis, info_idx, contents)
                r_loss_avg = 0
                p_loss_avg = 0
                c_loss_avg = 0
                f_loss_avg = 0
                loss_avg = 0
                info_idx += 1

            iteration += 1

        epoch += 1


def eval_on_dataset(config, data_loader, detail_model, context_model,
                    trans_len, debug=False, post_process=True):
    device = data_loader.dataset.device
    dtype = data_loader.dataset.dtype

    indices = config["indices"]
    context_len = config["train"]["context_len"]
    target_idx = context_len + trans_len
    seq_slice = slice(context_len, target_idx)
    window_len = context_len + trans_len + 2

    mean_ctx, std_ctx = ctx_mdl.get_train_stats_torch(config, dtype, device)
    mean_state, std_state, _, _ = get_train_stats_torch(
        config, dtype, device)
    mean_rmi, std_rmi = rmi.get_rmi_benchmark_stats_torch(
        config, dtype, device)

    # attention mask for context model
    atten_mask_ctx = ctx_mdl.get_attention_mask(
        window_len, context_len, target_idx, device)

    # attention mask for detail model
    atten_mask = get_attention_mask(window_len, target_idx, device)

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

        pos_new, rot_new, _ = evaluate(
            detail_model, context_model, positions, rotations, foot_contact,
            seq_slice, indices, mean_ctx, std_ctx, mean_state, std_state,
            atten_mask, atten_mask_ctx, post_process)

        (gpos_batch_loss, gquat_batch_loss,
         npss_batch_loss, npss_batch_weights) = get_rmi_style_batch_loss(
            positions, rotations, pos_new, rot_new, parents,
            context_len, target_idx, mean_rmi, std_rmi)

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


def evaluate(detail_model, context_model, positions, rotations, foot_contact,
             seq_slice, indices, mean_ctx, std_ctx, mean_state, std_state,
             atten_mask, atten_mask_ctx, post_process=True, midway_targets=()):
    dtype = positions.dtype
    device = positions.device
    window_len = positions.shape[-3]
    context_len = seq_slice.start
    target_idx = seq_slice.stop
    rp_slice = slice(indices["r_start_idx"], indices["p_end_idx"])
    c_slice = slice(indices["c_start_idx"], indices["c_end_idx"])

    if midway_targets:
        # If either context model or detail model is not trained with
        # constraints, ignore midway_targets.
        if not context_model.constrained_slices:
            print(
                "WARNING: Context model is not trained with constraints, but "
                "midway_targets is provided with values while calling evaluate()! "
                "midway_targets is ignored!"
            )
            midway_targets = []
        elif not detail_model.constrained_slices:
            print(
                "WARNING: Detail model is not trained with constraints, but "
                "midway_targets is provided with values while calling evaluate()! "
                "midway_targets is ignored!"
            )
            midway_targets = []

    with torch.no_grad():
        detail_model.eval()
        context_model.eval()

        # get context model output
        pos_ctx, rot_ctx = ctx_mdl.evaluate(
            context_model, positions, rotations, seq_slice,
            indices, mean_ctx, std_ctx, atten_mask_ctx,
            post_process=True, midway_targets=midway_targets)

        data_mask = ctx_mdl.get_data_mask(
            window_len, detail_model.d_mask,
            detail_model.constrained_slices, context_len, target_idx,
            device, dtype, midway_targets=midway_targets)

        state_in, delta_in = get_model_input(positions, rotations)
        state, delta = get_model_input(pos_ctx, rot_ctx)

        reset_constrained_values(
            state, state_in, delta, delta_in,
            midway_targets, detail_model.constrained_slices)

        state_zscore = (state - mean_state) / std_state

        # get detail model output
        state_out, c_out = get_model_output(
            detail_model, state_zscore, data_mask, atten_mask, foot_contact,
            seq_slice, c_slice, rp_slice)

        if post_process:
            state_out = train_utils.anim_post_process(
                state_out, state_zscore, seq_slice)

        state_out = state_out * std_state + mean_state

        pos_new = train_utils.get_new_positions(
            positions, state_out, indices, seq_slice)
        rot_new = train_utils.get_new_rotations(
            state_out, indices, rotations, seq_slice)

        return pos_new, rot_new, c_out
