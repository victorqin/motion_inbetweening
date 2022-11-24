import os
import pickle
import unittest

import numpy as np


tests_root = os.path.dirname(os.path.abspath(__file__))


class TestTrainStats(unittest.TestCase):
    CONFIG_NAME = "lafan1_context_model"

    def test_rmi_train_stats(self):
        from motion_inbetween.train import rmi
        from motion_inbetween.config import load_config_by_name

        config = load_config_by_name(self.CONFIG_NAME)
        mean, std = rmi.get_rmi_benchmark_stats(config, use_cache=False)

        # load training stats provided by Robust Motion Inbetweening paper
        ref_stats_path = os.path.join(tests_root, "train_stats_rmi.pkl")
        with open(ref_stats_path, "rb") as fh:
            ref_train_stats = pickle.load(fh)
            ref_mean = ref_train_stats["x_mean"]
            ref_std = ref_train_stats["x_std"]

        self.assertAlmostEqual(
            np.mean(np.abs(mean - ref_mean[0, :, 0])), 0, delta=5e-5)
        self.assertAlmostEqual(
            np.mean(np.abs(std - ref_std[0, :, 0])), 0, delta=5e-5)

    def test_context_train_stats(self):
        from motion_inbetween.train import context_model
        from motion_inbetween.config import load_config_by_name

        context_len = 10
        config = load_config_by_name(self.CONFIG_NAME)
        config["train"]["context_len"] = context_len
        mean, std = context_model.get_train_stats(config, use_cache=False)

        ref_stats_path = os.path.join(tests_root, "train_stats_context.pkl")
        with open(ref_stats_path, "rb") as fh:
            ref_train_stats = pickle.load(fh)
            ref_mean = ref_train_stats["mean"]
            ref_std = ref_train_stats["std"]

        self.assertAlmostEqual(np.mean(np.abs(mean - ref_mean)), 0)
        self.assertAlmostEqual(np.mean(np.abs(std - ref_std)), 0)
