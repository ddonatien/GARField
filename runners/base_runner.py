import os
import torch
import cv2 as cv
import numpy as np
from pyhocon import ConfigFactory
from models.dataset import MultiviewDataset


class BaseRunner:
    def __init__(
        self, conf_path, wdb_run, mode="train", case="CASE_NAME", is_continue=False
    ):
        self.device = torch.device("cuda")

        self.iter_step = 0

        self.is_continue = is_continue
        self.mode = mode

        # Configuration
        self.conf_path = conf_path
        with open(self.conf_path) as f:
            conf_text = f.read()
            conf_text = conf_text.replace("CASE_NAME", case)

        self.wdb_run = wdb_run
        self.continue_name = None

        self.conf = ConfigFactory.parse_string(conf_text)
        self.parse_cfg_general()
        self.parse_cfg_dataset(case)
        self.parse_cfg_train()

    def parse_cfg_general(self):
        self.base_exp_dir = self.conf["general.base_exp_dir"]
        if "general.base_run_dir" in self.conf:
            self.base_run_dir = self.conf["general.base_run_dir"]
        else:
            self.base_run_dir = "./"
        os.makedirs(self.base_exp_dir, exist_ok=True)

    def parse_cfg_dataset(self, case):
        self.conf["dataset.data_dir"] = self.conf["dataset.data_dir"].replace(
            "CASE_NAME", case
        )
        self.is_multiview = self.conf.get_bool("general.is_multiview")
        self.dataset = MultiviewDataset(self.conf["dataset"])
        self.use_depth = self.conf.get_bool("dataset.use_depth")

    def parse_cfg_train(self):
        self.end_iter = self.conf.get_int("train.end_iter")
        self.save_freq = self.conf.get_int("train.save_freq")
        self.max_pe_iter = self.conf.get_int("train.max_pe_iter")
        self.report_freq = self.conf.get_int("train.report_freq")
        self.val_freq = self.conf.get_int("train.val_freq")
        self.batch_size = self.conf.get_int("train.batch_size")
        self.validate_resolution_level = self.conf.get_int(
            "train.validate_resolution_level"
        )
        self.learning_rate = self.conf.get_float("train.learning_rate")
        self.learning_rate_alpha = self.conf.get_float("train.learning_rate_alpha")
        self.use_white_bkgd = self.conf.get_bool("train.use_white_bkgd")
        self.warm_up_end = self.conf.get_float("train.warm_up_end", default=0.0)
        self.anneal_end = self.conf.get_float("train.anneal_end", default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float("train.igr_weight")
        self.mask_weight = self.conf.get_float("train.mask_weight")

        if self.use_depth:
            self.geo_weight = self.conf.get_float("train.geo_weight")
        self.scene_components = self.conf["train.scene_components"]

    def train(self):
        raise NotImplementedError

    def log_slice_img(self, data_raw):
        data_raw_log = np.log10(1 + np.abs(data_raw))
        slice_log_img = (
            np.clip(data_raw_log / np.log10(np.sqrt(12)), a_max=1, a_min=0) * 255.0
        )
        slice_log_img = np.uint8(slice_log_img)
        slice_log_img = cv.applyColorMap(slice_log_img, cv.COLORMAP_JET)

        return slice_log_img

    def slice_img(self, data_raw):
        slice_img = np.clip((1 + data_raw) / np.sqrt(12), a_max=1, a_min=0) * 255.0
        slice_img = np.uint8(slice_img)
        slice_img = cv.applyColorMap(slice_img, cv.COLORMAP_JET)

        return slice_img
