import os
import json
from typing import List

import numpy as np
import torch
from torchvision.transforms import Compose
from sklearn.utils.class_weight import compute_class_weight


from ..augmentations import Translate, RandomRotation


class SlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: str = "./datasets/hri-is/coco17",
        spots_path: str = "./datasets/hri-is",
        window_size: int = 32,
        stride: int = 1,
        class_id_list: List[int] = [1, 15],
        background: bool = True,
        person_id_list: List[int] = [1],
        transform: bool = True,
        binary: bool = True,
    ):
        self.window_size = window_size
        self.background = background
        if transform:
            self.transform = Compose(
                [
                    Translate(),
                    RandomRotation(max_angle=0.3, axes=("z")),
                ]
            )
        else:
            self.transform = Compose(
                [
                    Translate(),
                ]
            )
        raw_data = self.load_all_sequences(
            dataset_path=dataset_path,
            spots_path=spots_path,
            person_id_list=person_id_list,
            class_id_list=class_id_list,
        )
        self._data, self._label = self.build_windows(
            data=raw_data,
            window_size=window_size,
            stride=stride,
            binary=binary,
        )

    def json_to_binary_vector(self, data):
        n_samples = data["n_samples"]
        gesture_ranges = data["labels"]
        binary_vector = np.zeros(n_samples, dtype=int)
        for g in gesture_ranges:
            binary_vector[g["begin"] : g["end"]] = 1
        return binary_vector

    def load_all_sequences(self, dataset_path, spots_path, person_id_list, class_id_list):
        data = []
        for person_id in person_id_list:
            for i, gesture_id in enumerate(class_id_list, start=1):
                filename = os.path.join(
                    dataset_path,
                    "p{:03d}g{:02d}_3d.npy".format(person_id, gesture_id),
                )
                spots_filename = os.path.join(
                    spots_path,
                    "p{:03d}g{:02d}_spots.json".format(person_id, gesture_id),
                )
                with open(spots_filename) as json_file:
                    spots_data = json.load(json_file)
                skeleton_video_np = np.load(
                    filename,
                    allow_pickle=True,
                )
                skeleton_video_np = torch.from_numpy(skeleton_video_np)
                skeleton_video_np = skeleton_video_np.permute(1, 0, 2, 3).float()  # T, M, V, C -> M, T, V, C
                data.append(
                    {
                        "video": skeleton_video_np,
                        "spots": self.json_to_binary_vector(spots_data) * i,
                    },
                )
        return data

    def build_windows(self, data, window_size=5, stride=1, binary=False):
        windows = []
        labels = []
        for item in data:
            spots = item["spots"]
            video = item["video"]
            length = video.shape[1]
            for start_idx in range(0, length, stride):
                if start_idx < window_size // 2:
                    sample_s = 0
                    sample_e = window_size
                elif start_idx > length - (window_size // 2):
                    sample_s = length - window_size
                    sample_e = length
                else:
                    sample_s = start_idx - (window_size // 2)
                    sample_e = sample_s + window_size

                window = video[:, sample_s:sample_e, :, :]  # shape [M, window_size, V, C]
                label = spots[(sample_s + sample_e) // 2]

                label_clip = spots[sample_s:sample_e]
                p_non_zero = np.count_nonzero(label_clip) / len(label_clip)
                if p_non_zero > 0.5 and label == 0:
                    counts = np.bincount(label_clip)
                    label = int(np.argmax(counts))

                if not self.background:
                    if label == 0:
                        continue
                    else:
                        label = label - 1
                else:
                    if label != 0 and binary:
                        label = 1
                windows.append(window)
                labels.append(label)

        return windows, labels

    def __getitem__(self, index):
        x = self._data[index]
        x = self.transform(x)
        return x, self._label[index]

    def __len__(self):
        return len(self._data)

    def compute_class_weights(self):
        weights = compute_class_weight(class_weight="balanced", classes=np.unique(self._label), y=self._label)
        return weights
