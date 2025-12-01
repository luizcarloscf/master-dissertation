from typing import List

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset

from .tools import random_rot, valid_crop_resize


class NTURGBD(Dataset):
    def __init__(
        self,
        data_path: str,
        p_interval: List[float] = 1.0,
        split: str = "train",
        random_rot: bool = False,
        window_size: int = -1,
        normalization: bool = False,
        use_mmap: bool = False,
    ):
        """
        data_path:
        label_path:
        split: training set or test set
        random_rot: rotate skeleton around xyz axis
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        use_mmap: If true, use mmap mode to load data, which can save the running memory
        only_label: only load label for ensemble score compute
        """
        self.data_path = data_path
        self.split = split
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        if self.use_mmap:
            npz_data = np.load(self.data_path, mmap_mode="r")
        else:
            npz_data = np.load(self.data_path)

        if self.split == "train":
            self.data = npz_data["x_train"]
            self.label = np.where(npz_data["y_train"] > 0)[1]
            self.sample_name = ["train_" + str(i) for i in range(len(self.data))]
        elif self.split == "test":
            self.data = npz_data["x_test"]
            self.label = np.where(npz_data["y_test"] > 0)[1]
            self.sample_name = ["test_" + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError("data split only supports train/test")

        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = random_rot(data_numpy)

        C, T, V, M = data_numpy.shape
        data_numpy = data_numpy.permute(3, 1, 2, 0)
        return data_numpy, label

    def compute_class_weights(self):
        # labels = [y for _, y in self]
        weights = compute_class_weight(class_weight="balanced", classes=np.unique(self.label), y=self.label)
        return weights
