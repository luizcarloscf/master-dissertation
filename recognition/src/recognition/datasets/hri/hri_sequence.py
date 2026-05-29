import os
import json
from typing import List

import numpy as np
import torch


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: str = "./datasets/hri-is/coco17",
        spots_path: str = "./datasets/hri-is",
        class_id_list: List[int] = [1, 15],
        person_id_list: List[int] = [1],
        binary: bool = True,
    ):
        self.binary = binary
        self.data = self.load_all_sequences(
            dataset_path=dataset_path,
            spots_path=spots_path,
            person_id_list=person_id_list,
            class_id_list=class_id_list,
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
                        "filename": "p{:03d}g{:02d}".format(person_id, gesture_id),
                        "video": skeleton_video_np,
                        "spots": self.json_to_binary_vector(spots_data) if self.binary else self.json_to_binary_vector(spots_data) * i,
                    },
                )
        return data

    def __getitem__(self, index):
        x = self.data[index]["video"]
        return x, self.data[index]["spots"], self.data[index]["filename"]

    def __len__(self):
        return len(self.data)
