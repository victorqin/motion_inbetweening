import torch
import numpy as np

from motion_inbetween.data import utils_torch as data_utils


"""
Note:
In order to get the same loss metric as Robust Motion Inbetweening, we should:
1) Use mean centered clip data (data_utils.to_mean_centered_data())
2) Get mean and std of global position (mean_rmi, std_rmi = rmi.get_rmi_benchmark_stats_torch())
3) For global pos loss, apply zscore normalization before calculating the loss
For global quaternion loss, no need to apply zscore.
"""


def get_l2loss_batch(gt_data, pred_data):
    """
    Get average l2 loss of each seq (per frame average) of every batch.

    Args:
        gt_data (tensor): (batch, seq, dim)
        pred_data (tensor): (batch, seq, dim)

    Returns:
        np.array: l2 loss, shape: (batch, )
    """
    # (batch, seq, dim)
    delta = torch.square(gt_data - pred_data)

    # (batch, seq)
    delta = torch.sqrt(torch.sum(delta, dim=-1))

    # (batch, )
    delta = torch.mean(delta, dim=-1)

    return delta.cpu().numpy()


def get_npss_loss_batch(gt_data, pred_data, eps=1e-8):
    """
    Computes Normalized Power Spectrum Similarity (NPSS).

    This is the metric proposed by Gropalakrishnan et al (2019).

    Args:
        gt_data (tensor): (batch, seq, dim)
        pred_data (tensor): (batch, seq, dim)
        eps (float, optional): Small value to avoid division by zero.
            Defaults to 1e-8.
    """
    gt_data = gt_data.cpu().numpy()
    pred_data = pred_data.cpu().numpy()

    # Fourier coefficients along the time dimension, (batch, seq, dim)
    gt_fourier_coeffs = np.real(np.fft.fft(gt_data, axis=1))
    pred_fourier_coeffs = np.real(np.fft.fft(pred_data, axis=1))

    # Square of the Fourier coefficients, (batch, seq, dim)
    gt_power = np.square(gt_fourier_coeffs)
    pred_power = np.square(pred_fourier_coeffs)

    # Sum of powers over time dimension, (batch, dim)
    gt_total_power = np.sum(gt_power, axis=-2)
    pred_total_power = np.sum(pred_power, axis=-2)

    # avoid division by zero
    gt_total_power_tmp = gt_total_power[:, np.newaxis, :].copy()
    gt_total_power_tmp[gt_total_power_tmp < eps] = np.inf
    pred_total_power_tmp = pred_total_power[:, np.newaxis, :].copy()
    pred_total_power_tmp[pred_total_power_tmp < eps] = np.inf

    # Normalize powers with totals, (batch, seq, dim)
    gt_norm_power = gt_power / gt_total_power_tmp
    pred_norm_power = pred_power / pred_total_power_tmp

    # Cumulative sum over time, (batch, seq, dim)
    cdf_gt_power = np.cumsum(gt_norm_power, axis=-2)
    cdf_pred_power = np.cumsum(pred_norm_power, axis=-2)

    # Earth mover distance, (batch, dim)
    emd = np.linalg.norm((cdf_pred_power - cdf_gt_power), ord=1, axis=-2)

    # Weighted EMD, (batch, )
    # This is the original code from Robust Motion Inbetweening.
    # power_weighted_emd = np.average(emd, weights=gt_total_power, axis=-1)

    # emd: (batch, dim),  emd weights: (batch, dim)
    return emd, gt_total_power


def get_rmi_style_batch_loss(positions, rotations, pos_new, rot_new,
                             parents, context_len, target_idx,
                             mean_rmi, std_rmi):
    seq_slice = slice(context_len, target_idx)

    # Convert to mean centered data in order to be compatible
    # to Robust Motion Inbetween metrics.

    # Ground Truth ---------------------------------------------------------
    positions, rotations, root_pos_offset, root_rot_offset = \
        data_utils.to_mean_centered_data(positions, rotations, context_len,
                                         return_offset=True)
    global_rotations, global_positions = data_utils.fk_torch(
        rotations, positions, parents)

    # gpos_zscore: (batch, seq, joint, 3)
    # gquat: (batch, seq, joint, 4)
    gpos_zscore = (global_positions - mean_rmi) / std_rmi
    gquat = data_utils.matrix9D_to_quat_torch(global_rotations)
    gquat = data_utils.remove_quat_discontinuities(gquat)

    # Predicted ------------------------------------------------------------
    pos_new, rot_new = data_utils.apply_root_pos_rot_offset(
        pos_new, rot_new, root_pos_offset, root_rot_offset)
    grot_new, gpos_new = data_utils.fk_torch(rot_new, pos_new, parents)

    gpos_new_zscore = (gpos_new - mean_rmi) / std_rmi
    gquat_new = data_utils.matrix9D_to_quat_torch(grot_new)
    gquat_new = data_utils.remove_quat_discontinuities(gquat_new)

    # Loss -----------------------------------------------------------------
    gpos_batch_loss = get_l2loss_batch(
        gpos_zscore[..., seq_slice, :, :].flatten(-2),
        gpos_new_zscore[..., seq_slice, :, :].flatten(-2))
    gquat_batch_loss = get_l2loss_batch(
        gquat[..., seq_slice, :, :].flatten(-2),
        gquat_new[..., seq_slice, :, :].flatten(-2))
    npss_batch_loss, npss_batch_weight = get_npss_loss_batch(
        gquat[..., seq_slice, :, :].flatten(-2),
        gquat_new[..., seq_slice, :, :].flatten(-2)
    )

    return gpos_batch_loss, gquat_batch_loss, npss_batch_loss, npss_batch_weight


# zero velocity baseline ####################################################
def get_zerov_loss(dataset, data_loader, context_len, target_idx,
                   mean_rmi, std_rmi, debug=False):
    """
    Calculate global position and quationion loss for zero velocity baseline.
    """
    seq_slice = slice(context_len, target_idx)
    ctx_slice = slice(context_len - 1, context_len)

    data_indexes = []
    gpos_loss = []
    gquat_loss = []
    npss_loss = []
    npss_weights = []

    for i, data in enumerate(data_loader, 0):
        (positions, rotations, _, _, _, parents, data_idx) = data
        parents = parents[0]

        positions, rotations = data_utils.to_mean_centered_data(
            positions, rotations, context_len)

        pos_zerov = positions.clone().detach()
        pos_zerov[:, seq_slice] = positions[:, ctx_slice]
        rot_zerov = rotations.clone().detach()
        rot_zerov[:, seq_slice] = rotations[:, ctx_slice]

        (gpos_batch_loss, gquat_batch_loss,
         npss_batch_loss, npss_batch_weights) = get_rmi_style_batch_loss(
            positions, rotations, pos_zerov, rot_zerov, parents,
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


# interpolation baseline ####################################################
def get_interp_loss(dataset, data_loader, context_len, target_idx,
                    mean_rmi, std_rmi, debug=False):
    """
    Calculate global position and quationion loss for interpolation baseline.
    """
    seq_slice = slice(context_len, target_idx)

    data_indexes = []
    gpos_loss = []
    gquat_loss = []
    npss_loss = []
    npss_weights = []

    for i, data in enumerate(data_loader, 0):
        (positions, rotations, _, _, _, parents, data_idx) = data
        parents = parents[0]

        positions, rotations = data_utils.to_mean_centered_data(
            positions, rotations, context_len)

        # interpolated
        pos_interp, rot_interp = get_interpolated_local_pos_rot(
            positions, rotations, seq_slice)

        (gpos_batch_loss, gquat_batch_loss,
         npss_batch_loss, npss_batch_weights) = get_rmi_style_batch_loss(
            positions, rotations, pos_interp, rot_interp, parents,
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


def get_interpolated_local_pos_rot(positions, rotations, seq_slice):
    """
    Get interpolated global position and rotations.

    Args:
        positions (tensor): local position sequence.
            Shape: (batch, seq, joint, 3)
        rotations (tensor): local rotation sequence.
            Shape: (batch, seq, joint, 3, 3)
        seq_slice (slice): Sequence slice of output data.

    Returns:
        tensor, tensor: (inter_local_pos, inter_local_rot)
            Shape: (batch, seq, joint, 3),
                   (batch, seq, joint, 4, 4)
    """
    local_pos, local_quat = _get_interpolated_local_pos_quat(
        positions, rotations, seq_slice)
    local_rot = data_utils.quat_to_matrix9D_torch(local_quat)

    inter_local_pos = positions.clone().detach()
    inter_local_pos[..., seq_slice, :, :] = local_pos
    inter_local_rot = rotations.clone().detach()
    inter_local_rot[..., seq_slice, :, :, :] = local_rot

    return inter_local_pos, inter_local_rot


def get_linear_interpolation(start, end, length):
    """
    Get linear interpolation between start and end.

    Args:
        start (tensor): shape: (batch, 1, dim)
        end (tensor): shape: (batch, 1, dim)
        length (int): length of output

    Returns:
        tensor: (batch, length, dim)
    """
    offset = end - start            # (batch, 1, dim)
    t = torch.linspace(0.0, 1.0, steps=length + 2,
                       dtype=start.dtype, device=start.device)
    t = t[..., None]                # (length+2, 1)

    output = start + t * offset     # (batch, length+2, dim)
    return output[..., 1:-1, :]     # (batch, length, dim)


def _get_interpolated_local_pos_quat(positions, rotations, seq_slice):
    context_slice = slice(seq_slice.start - 1, seq_slice.start)
    target_slice = slice(seq_slice.stop, seq_slice.stop + 1)
    n_trans = seq_slice.stop - seq_slice.start

    # Lerp root position
    root_pos = get_linear_interpolation(
        positions[..., context_slice, 0, :],
        positions[..., target_slice, 0, :], n_trans)[..., None, :]

    pos = torch.cat([root_pos, positions[..., seq_slice, 1:, :]], dim=-2)

    # Slerp local quaterions
    gt_quat = data_utils.matrix9D_to_quat_torch(rotations)
    gt_quat = data_utils.remove_quat_discontinuities(gt_quat)
    start_quat = gt_quat[..., context_slice, :, :]
    end_quat = gt_quat[..., target_slice, :, :]

    t = torch.linspace(0.0, 1.0, steps=n_trans + 2,
                       dtype=positions.dtype, device=positions.device)
    quat = [
        data_utils.normalize_torch(data_utils.quat_slerp_torch(
            start_quat, end_quat, i)) for i in t
    ]
    quat = torch.cat(quat, dim=-3)
    quat = quat[..., 1: -1, :, :]

    return pos, quat
