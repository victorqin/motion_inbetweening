import os
import torch
from torch.utils.data import Dataset
import numpy as np

from motion_inbetween.data import bvh, utils_np


class BvhDataSet(Dataset):
    def __init__(self, bvh_folder, actors, window=50, offset=1,
                 start_frame=0, device="cpu", dtype=torch.float32):
        """
        Bvh data set.

        Args:
            bvh_folder (str): Bvh folder path.
            actors (list of str): List of actors to be included in the dataset.
            window (int, optional): Length of window. Defaults to 50.
            offset (int, optional): Offset of window. Defaults to 1.
            start_frame (int, optional):
                Override the start frame of each bvh file. Defaults to 0.
            device (str, optional): Device. e.g. "cpu", "cuda:0".
                Defaults to "cpu".
            dtype: torch.float16, torch.float32, torch.float64 etc.
        """
        super(BvhDataSet, self).__init__()
        self.bvh_folder = bvh_folder
        self.actors = actors
        self.window = window
        self.offset = offset
        self.start_frame = start_frame
        self.device = device
        self.dtype = dtype

        self.load_bvh_files()

    def _to_tensor(self, array):
        return torch.tensor(array, dtype=self.dtype, device=self.device)

    def load_bvh_files(self):
        self.bvh_files = []
        self.positions = []
        self.rotations = []
        self.global_positions = []
        self.global_rotations = []
        self.foot_contact = []
        self.frames = []
        self.parents = []

        # load bvh files that match given actors
        for f in os.listdir(self.bvh_folder):
            f = os.path.abspath(os.path.join(self.bvh_folder, f))
            if f.endswith(".bvh"):
                file_name = os.path.basename(f).rsplit(".", 1)[0]
                seq_name, actor = file_name.rsplit("_", 1)
                if actor in self.actors:
                    self.bvh_files.append(f)

        if not self.bvh_files:
            raise FileNotFoundError(
                "No bvh files found in {}. (Actors: {})".format(
                    self.bvh_folder, ", ".join(self.actors))
            )

        self.bvh_files.sort()
        for bvh_path in self.bvh_files:
            print("Processing file {}".format(bvh_path))
            anim = bvh.load_bvh(bvh_path, start=self.start_frame)

            # global joint rotation, position
            gr, gp = utils_np.fk(anim.rotations, anim.positions, anim.parents)

            # left, right foot contact
            cl, cr = utils_np.extract_feet_contacts(
                gp, [3, 4], [7, 8], vel_threshold=0.2)

            self.positions.append(self._to_tensor(anim.positions))
            self.rotations.append(self._to_tensor(anim.rotations))
            self.global_positions.append(self._to_tensor(gp))
            self.global_rotations.append(self._to_tensor(gr))
            self.foot_contact.append(self._to_tensor(
                np.concatenate([cl, cr], axis=-1)))
            self.frames.append(anim.positions.shape[0])
            self.parents = anim.parents

    def __len__(self):
        count = 0
        for frame in self.frames:
            count += int(float(frame - self.window) / self.offset) + 1
        return count

    def __getitem__(self, idx):
        curr_idx = idx

        for i, frame in enumerate(self.frames):
            tmp_idx = curr_idx - \
                int(float(frame - self.window) / self.offset) - 1

            if tmp_idx >= 0:
                curr_idx = tmp_idx
                continue

            start_idx = curr_idx * self.offset
            end_idx = start_idx + self.window

            # print(idx, i, start_idx, end_idx)

            positions = self.positions[i][start_idx: end_idx]
            rotations = self.rotations[i][start_idx: end_idx]
            global_positions = self.global_positions[i][start_idx: end_idx]
            global_rotations = self.global_rotations[i][start_idx: end_idx]
            foot_contact = self.foot_contact[i][start_idx: end_idx]

            return (
                positions,
                rotations,
                global_positions,
                global_rotations,
                foot_contact,
                self.parents,
                idx
            )
