import os
import json
from typing import List

import numpy as np
import torch
from torchvision.transforms import Compose
from sklearn.utils.class_weight import compute_class_weight

from .augmentations import TranslateToOrigin, RandomRotation, ValidCropResize


class ClipDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        p_interval: List[int],
        dataset_path: str = "./datasets/hri-is/coco17",
        spots_path: str = "./datasets/hri-is/",
        num_frames: int = 64,
        person_id_list: List[int] = [1, 2, 3, 4, 5],
        class_id_list: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        transform: bool = True,
        background: bool = True,
        repeat: int = 1,
        min_background_samples: int = 16,
    ):
        self.repeat = repeat
        self.p_interval = p_interval
        self.num_frames = num_frames
        self.background = background
        self.min_background_samples = min_background_samples
        if transform:
            self.transform = Compose(
                [
                    TranslateToOrigin(joint_idx=0),
                    ValidCropResize(p_interval=p_interval, window=num_frames, min_size=32),
                    RandomRotation(max_angle=0.3, axes=("z")),
                ]
            )
        else:
            self.transform = Compose(
                [
                    ValidCropResize(p_interval=p_interval, window=num_frames, min_size=32),
                    TranslateToOrigin(joint_idx=0),
                ]
            )
        raw_data = self.load_all_sequences(
            dataset_path=dataset_path,
            spots_path=spots_path,
            person_id_list=person_id_list,
            class_id_list=class_id_list,
        )
        self._data, self._label = self.make_clips(data=raw_data)

    def convert_segments(self, data, gesture_class, non_gesture_class=0):
        n_samples = data["n_samples"]
        labels = data["labels"]
        labels = sorted(labels, key=lambda x: x["begin"])
        new_labels = []
        prev_end = 0
        for seg in labels:
            begin, end = seg["begin"], seg["end"]
            if begin > prev_end:
                new_labels.append({"begin": prev_end, "end": begin, "class": non_gesture_class})
            new_labels.append({"begin": begin, "end": end, "class": gesture_class})
            prev_end = end
        if prev_end < n_samples:
            new_labels.append({"begin": prev_end, "end": n_samples, "class": non_gesture_class})

        return {"n_samples": n_samples, "labels": new_labels}

    def load_all_sequences(self, dataset_path, spots_path, person_id_list, class_id_list):
        data = []
        for person_id in person_id_list:
            for gesture_id in class_id_list:
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
                if self.background:
                    spots_data = self.convert_segments(spots_data, gesture_class=gesture_id, non_gesture_class=0)

                skeleton_video_np = np.load(
                    filename,
                    allow_pickle=True,
                )
                skeleton_video_np = torch.from_numpy(skeleton_video_np)
                skeleton_video_np = skeleton_video_np.permute(1, 0, 2, 3).float()  # T, M, V, C -> M, T, V, C
                data.append(
                    {
                        "video": skeleton_video_np,
                        "spots": spots_data,
                        "label": gesture_id,
                    },
                )
        return data

    def make_clips(self, data):
        clips = []
        labels = []
        for item in data:
            video = item["video"]
            gesture_ranges = item["spots"]["labels"]
            for g in gesture_ranges:
                if self.background:
                    if g["class"] == 0 and ((g["end"] - g["begin"]) < self.min_background_samples):
                        continue
                    clips.append(video[:, g["begin"] : g["end"]])
                    labels.append(g["class"])
                else:
                    clips.append(video[:, g["begin"] : g["end"]])
                    labels.append(item["label"] - 1)
        return clips, labels

    def repeat_to_lenght(self, x, length):
        M, T, V, C = x.shape
        repeat_times = (length + x.shape[1] - 1) // x.shape[1]
        x_repeated = x.repeat((1, repeat_times, 1, 1))[:, :length]
        return x_repeated

    def __getitem__(self, index):
        x = self._data[index % len(self._data)]
        x = self.transform(x)
        return x, self._label[index % len(self._label)]

    def __len__(self):
        return len(self._data) * self.repeat

    def compute_class_weights(self):
        # labels = [y for _, y in self]
        weights = compute_class_weight(class_weight="balanced", classes=np.unique(self._label), y=self._label)
        return weights
