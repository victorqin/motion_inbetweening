import os
import unittest
import numpy as np

import torch
from torch.utils.data import DataLoader

from motion_inbetween import DATASET_ROOT
from motion_inbetween.data import bvh
from motion_inbetween.data import utils_np, utils_torch
from motion_inbetween.data import loader


anim_path = os.path.join(DATASET_ROOT, "lafan1", "walk1_subject1.bvh")
anim = bvh.load_bvh(anim_path)


class TestBvh(unittest.TestCase):
    def test_bvh_loader(self):
        anim_path = os.path.join(DATASET_ROOT, "lafan1", "walk1_subject1.bvh")

        anim1 = bvh.load_bvh(anim_path)
        self.assertEqual(anim1.rotations.shape, (7840, 22, 3, 3))
        self.assertEqual(anim1.positions.shape, (7840, 22, 3))
        self.assertEqual(anim1.offsets.shape, (22, 3))
        self.assertEqual(anim1.parents.shape, (22, ))
        self.assertEqual(len(anim1.names), 22)

        start_frame = 10
        anim2 = bvh.load_bvh(anim_path, start=start_frame)
        self.assertEqual(anim1.rotations.shape[0],
                         anim2.rotations.shape[0] + start_frame)

        self.assertAlmostEqual(
            np.sum(np.abs(anim1.rotations[start_frame:] - anim2.rotations)),
            0)
        self.assertAlmostEqual(
            np.sum(np.abs(anim1.positions[start_frame:] - anim2.positions)),
            0)


class TestDataUtilsNumpy(unittest.TestCase):
    def test_6D_9D_rotation_conversion(self):
        rot = anim.rotations
        rot_6d = utils_np.matrix9D_to_6D(rot)
        rot_9d = utils_np.matrix6D_to_9D(rot_6d)
        delta = rot - rot_9d

        self.assertEqual(rot_6d.shape[:-1], rot_9d.shape[:-2])
        self.assertEqual(rot_6d.shape[-1], 6)
        self.assertEqual(rot_9d.shape[-2:], (3, 3))
        self.assertAlmostEqual(np.sum(np.abs(delta)), 0)

        mat = np.random.rand(2, 4, 10, 6)       # (batch, seq, joint, dim)
        mat = utils_np.matrix6D_to_9D(mat)
        delta = np.matmul(mat, mat.swapaxes(-1, -2)) - np.identity(3)
        self.assertEqual(mat.shape, (2, 4, 10, 3, 3))
        self.assertAlmostEqual(np.sum(np.abs(delta)), 0)

    def test_root_vel(self):
        root_idx = 0
        frame_num = anim.positions.shape[0]

        # root_vel is padded with zero at first frame
        root_vel = utils_np.extract_root_vel(anim.positions, root_idx=0)

        root_pos = []
        current_pos = anim.positions[0, root_idx]
        for i in range(0, frame_num):
            current_pos = current_pos + root_vel[i]
            root_pos.append(current_pos)
        root_pos = np.array(root_pos)

        self.assertAlmostEqual(
            np.sum(np.abs(anim.positions[:, root_idx] - root_pos)), 0)

    def test_root_rot_vel(self):
        root_idx = 0
        frame_num = anim.positions.shape[0]

        root_rot_vel = utils_np.extract_root_rot_vel(
            anim.rotations, root_idx=root_idx)
        root_rot_vel_simple = utils_np.extract_root_rot_vel_simple(
            anim.rotations, root_idx=root_idx)

        root_rot = []
        root_rot_simple = []
        current_rot = anim.rotations[0, root_idx]
        current_rot_simple = anim.rotations[0, root_idx]
        for i in range(0, frame_num):
            current_rot = np.matmul(root_rot_vel[i], current_rot)
            current_rot_simple = root_rot_vel_simple[i] + current_rot_simple

            root_rot.append(current_rot)
            root_rot_simple.append(current_rot_simple)

        root_rot = np.array(root_rot)
        root_rot_simple = np.array(root_rot_simple)

        self.assertAlmostEqual(
            np.sum(np.abs(anim.rotations[:, root_idx] - root_rot)), 0)
        self.assertAlmostEqual(
            np.sum(np.abs(anim.rotations[:, root_idx] - root_rot_simple)), 0)


class TestDataUtilsTorch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # setup test dataset
        batch = 32
        window = 65
        offset = 50
        start_frame = 100
        bvh_folder = os.path.join(DATASET_ROOT, "lafan1")
        actors = ["subject1"]

        cls.dataset = loader.BvhDataSet(
            bvh_folder, actors, window, offset, start_frame,
            dtype=torch.float64)
        cls.data_loader = DataLoader(
            cls.dataset, batch_size=batch, shuffle=True)
        cls.joint = 22
        cls.batch = batch
        cls.window = window

    def test_6D_9D_rotation_conversion(self):
        rot = torch.tensor(anim.rotations)
        rot_6d = utils_torch.matrix9D_to_6D_torch(rot)
        rot_9d = utils_torch.matrix6D_to_9D_torch(rot_6d)
        delta = rot - rot_9d
        self.assertEqual(rot_6d.shape[:-1], rot_9d.shape[:-2])
        self.assertEqual(rot_6d.shape[-1], 6)
        self.assertEqual(rot_9d.shape[-2:], (3, 3))
        self.assertAlmostEqual(torch.sum(torch.abs(delta)).item(), 0)

        mat = torch.randn(2, 4, 10, 6).double()     # (batch, seq, joint, dim)
        mat = utils_torch.matrix6D_to_9D_torch(mat)
        delta = torch.matmul(mat, mat.transpose(-2, -1)) - torch.eye(3)
        self.assertEqual(mat.shape, (2, 4, 10, 3, 3))
        self.assertAlmostEqual(torch.sum(torch.abs(delta)).item(), 0)

    def test_start_and_mean_centered_data(self):
        context_len = 10
        target_idx = self.window - 1
        seq_slice = slice(context_len, target_idx)
        context_slice = slice(0, context_len)
        target_slice = slice(target_idx, self.window)

        for i, data in enumerate(self.data_loader, 0):
            (positions, rotations, global_positions, global_rotations,
             foot_contact, parents, data_idx) = data

            pos_start1, rot_start1 = utils_torch.to_start_centered_data(
                positions, rotations, context_len)
            pos_mean1, rot_mean1 = utils_torch.to_mean_centered_data(
                positions, rotations, context_len)

            pos_start2, rot_start2 = utils_torch.to_start_centered_data(
                pos_mean1, rot_mean1, context_len)
            pos_mean2, rot_mean2 = utils_torch.to_mean_centered_data(
                pos_start1, rot_start1, context_len)

            self.assertAlmostEqual(
                torch.sum(torch.abs(pos_start1 - pos_start2)).item(), 0)
            self.assertAlmostEqual(
                torch.sum(torch.abs(rot_start1 - rot_start2)).item(), 0)

            self.assertAlmostEqual(
                torch.sum(torch.abs(pos_mean1 - pos_mean2)).item(), 0)
            self.assertAlmostEqual(
                torch.sum(torch.abs(rot_mean1 - rot_mean2)).item(), 0)

            # modify part of data then apply conversion
            pos_new = positions.clone().detach()
            pos_new[..., seq_slice, :, :] = 0

            rot_new = rotations.clone().detach()
            rot_new[..., seq_slice, :, :, :] = \
                torch.eye(3, dtype=positions.dtype, device=positions.device)

            pos_start3, rot_start3, pos_start_offset3, rot_start_offset3 = \
                utils_torch.to_start_centered_data(
                    positions, rotations, context_len, return_offset=True)
            pos_mean3, rot_mean3, pos_mean_offset3, rot_mean_offset3 = \
                utils_torch.to_mean_centered_data(
                    positions, rotations, context_len, return_offset=True)

            pos_start4, rot_start4 = utils_torch.apply_root_pos_rot_offset(
                pos_new, rot_new, pos_start_offset3, rot_start_offset3)
            pos_mean4, rot_mean4 = utils_torch.apply_root_pos_rot_offset(
                pos_new, rot_new, pos_mean_offset3, rot_mean_offset3)

            pos_start_delta = torch.abs(pos_start3 - pos_start4)
            rot_start_delta = torch.abs(rot_start3 - rot_start4)
            pos_mean_delta = torch.abs(pos_mean3 - pos_mean4)
            rot_mean_delta = torch.abs(rot_mean3 - rot_mean4)

            self.assertAlmostEqual(
                torch.sum(pos_start_delta[:, context_slice]).item(), 0)
            self.assertAlmostEqual(
                torch.sum(pos_start_delta[:, target_slice]).item(), 0)
            self.assertAlmostEqual(
                torch.sum(rot_start_delta[:, context_slice]).item(), 0)
            self.assertAlmostEqual(
                torch.sum(rot_start_delta[:, target_slice]).item(), 0)

            self.assertAlmostEqual(
                torch.sum(pos_mean_delta[:, context_slice]).item(), 0)
            self.assertAlmostEqual(
                torch.sum(pos_mean_delta[:, target_slice]).item(), 0)
            self.assertAlmostEqual(
                torch.sum(rot_mean_delta[:, context_slice]).item(), 0)
            self.assertAlmostEqual(
                torch.sum(rot_mean_delta[:, target_slice]).item(), 0)

            # apply offset and reverse it
            pos_start5, rot_start5, pos_start_offset5, rot_start_offset5 = \
                utils_torch.to_start_centered_data(
                    positions, rotations, context_len, return_offset=True)
            pos_mean5, rot_mean5, pos_mean_offset5, rot_mean_offset5 = \
                utils_torch.to_mean_centered_data(
                    positions, rotations, context_len, return_offset=True)

            pos_start6, rot_start6 = utils_torch.reverse_root_pos_rot_offset(
                pos_start5, rot_start5, pos_start_offset5, rot_start_offset5)
            pos_mean6, rot_mean6 = utils_torch.reverse_root_pos_rot_offset(
                pos_mean5, rot_mean5, pos_mean_offset5, rot_mean_offset5)

            self.assertAlmostEqual(
                torch.sum(torch.abs(positions - pos_start6)).item(), 0)
            self.assertAlmostEqual(
                torch.sum(torch.abs(rotations - rot_start6)).item(), 0)
            self.assertAlmostEqual(
                torch.sum(torch.abs(positions - pos_mean6)).item(), 0)
            self.assertAlmostEqual(
                torch.sum(torch.abs(rotations - rot_mean6)).item(), 0)

            break

    def test_lafan1_dataset(self):
        for i, data in enumerate(self.data_loader, 0):
            (positions, rotations, global_positions, global_rotations,
             foot_contact, parents, data_idx) = data

            self.assertEqual(positions.shape,
                             (self.batch, self.window, self.joint, 3))
            self.assertEqual(rotations.shape,
                             (self.batch, self.window, self.joint, 3, 3))
            self.assertEqual(global_positions.shape,
                             (self.batch, self.window, self.joint, 3))
            self.assertEqual(global_rotations.shape,
                             (self.batch, self.window, self.joint, 3, 3))
            self.assertEqual(foot_contact.shape, (self.batch, self.window, 4))
            self.assertEqual(parents.shape, (self.batch, self.joint))

            break
