import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AddGaussianNoise(nn.Module):
    def __init__(self, scale: float = 0.1):
        super().__init__()
        assert 0 <= scale <= 1
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.std(x, dim=1, keepdim=True) * self.scale
        noise = (std**0.5) * torch.randn_like(x)
        x = x + noise
        return x


class TranslateToOrigin(nn.Module):
    def __init__(self, joint_idx: int):
        super().__init__()
        self.joint_idx = joint_idx

    def forward(self, skeleton_seq: torch.Tensor):
        M, T, V, C = skeleton_seq.shape
        ref_xy = skeleton_seq[:, 0, self.joint_idx, :2].unsqueeze(1).unsqueeze(2)  # [M, 1, 1, C]

        skeleton_seq_xy = skeleton_seq[..., :2] - ref_xy  # (M, T, V, 2)
        skeleton_seq_z = skeleton_seq[..., 2:].clone()  # (M, T, V, 1)

        return torch.cat((skeleton_seq_xy, skeleton_seq_z), dim=-1)  # (M, T, V, 3)


class Translate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        if len(x.shape) > 4:
            N, M, T, V, C = x.shape
            x = x.reshape(N * M, T, V, C)
            x_center = x[:, 0].mean(dim=1, keepdim=True).unsqueeze(1)  # [M, 1, 1, 3]
            x_out = x - x_center  # broadcast: [M, T, V, 3] - [M, 1, 1, 3]
            x_out = x_out.reshape(N, M, T, V, C)
            return x_out

        M, T, V, C = x.shape
        x_center = x[:, 0].mean(dim=1, keepdim=True).unsqueeze(1)  # [M, 1, 1, 3]
        x_out = x - x_center  # broadcast: [M, T, V, 3] - [M, 1, 1, 3]
        return x_out


class RandomScaling(nn.Module):
    def __init__(self, min_scale=0.9, max_scale=1.1):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self, x: torch.Tensor):
        M, T, V, C = x.shape
        scale = torch.empty(1, device=x.device, dtype=x.dtype).uniform_(self.min_scale, self.max_scale)
        return x * scale


class RandomRotation(nn.Module):
    def __init__(self, max_angle=0.3, axes=("x", "y", "z")):
        super().__init__()
        self.max_angle = max_angle
        self.axes = axes

    def _compute_rot_matrix(self, rot):
        cos_r, sin_r = rot.cos(), rot.sin()
        zeros = torch.zeros(rot.shape[0], 1)
        ones = torch.ones(rot.shape[0], 1)

        r1 = torch.stack((ones, zeros, zeros), dim=-1)
        rx2 = torch.stack((zeros, cos_r[:, 0:1], sin_r[:, 0:1]), dim=-1)
        rx3 = torch.stack((zeros, -sin_r[:, 0:1], cos_r[:, 0:1]), dim=-1)
        rx = torch.cat((r1, rx2, rx3), dim=1)  # [T, 3, 3]

        ry1 = torch.stack((cos_r[:, 1:2], zeros, -sin_r[:, 1:2]), dim=-1)
        r2 = torch.stack((zeros, ones, zeros), dim=-1)
        ry3 = torch.stack((sin_r[:, 1:2], zeros, cos_r[:, 1:2]), dim=-1)
        ry = torch.cat((ry1, r2, ry3), dim=1)

        rz1 = torch.stack((cos_r[:, 2:3], sin_r[:, 2:3], zeros), dim=-1)
        r3 = torch.stack((zeros, zeros, ones), dim=-1)
        rz2 = torch.stack((-sin_r[:, 2:3], cos_r[:, 2:3], zeros), dim=-1)
        rz = torch.cat((rz1, rz2, r3), dim=1)

        return rz @ ry @ rx  # [T, 3, 3]

    def _sample_angles(self, T):
        angles = torch.zeros(3)
        if "x" in self.axes:
            angles[0] = torch.empty(1).uniform_(-self.max_angle, self.max_angle)
        if "y" in self.axes:
            angles[1] = torch.empty(1).uniform_(-self.max_angle, self.max_angle)
        if "z" in self.axes:
            angles[2] = torch.empty(1).uniform_(-self.max_angle, self.max_angle)
        rot = angles.repeat(T, 1)  # [T, 3]
        return self._compute_rot_matrix(rot)  # [T, 3, 3]

    def forward(self, data):
        M, T, V, C = data.shape
        data = data.permute(1, 3, 2, 0).contiguous().view(T, C, V * M)  # [T, 3, V*M]
        rot_angles = self._sample_angles(T)  # [T, 3, 3]
        rotated = torch.matmul(rot_angles.to(data.dtype), data)  # [T, 3, V*M]
        rotated = rotated.view(T, C, V, M).permute(3, 0, 2, 1).contiguous()  # [M, T, V, C]
        return rotated


class AlignShouldersToXAxis(nn.Module):
    def __init__(self, left_shoulder: int, right_shoulder: int):
        super().__init__()
        self.left_shoulder = left_shoulder
        self.right_shoulder = right_shoulder

    def _align_single_sequence(self, seq: torch.Tensor):

        p_left = seq[0, self.left_shoulder]
        p_right = seq[0, self.right_shoulder]
        shoulder_vec = p_right - p_left
        shoulder_vec[2] = 0.0

        if torch.norm(shoulder_vec) < 1e-6:
            return seq

        shoulder_vec = F.normalize(shoulder_vec, dim=0)
        target_vec = torch.tensor([1.0, 0.0, 0.0], device=seq.device, dtype=shoulder_vec.dtype)

        axis = torch.cross(shoulder_vec, target_vec)
        angle = torch.acos(torch.clamp(torch.dot(shoulder_vec, target_vec), -1.0, 1.0))

        if torch.norm(axis) < 1e-6 or torch.isnan(angle):
            return seq

        axis = F.normalize(axis, dim=0)

        K = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], device=seq.device)

        I = torch.eye(3, device=seq.device)
        R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)

        T, V, C = seq.shape
        seq_flat = seq.view(-1, 3)  # [(T*V), 3]
        rotated = torch.matmul(seq_flat, R.T)  # [(T*V), 3]
        return rotated.view(T, V, C)

    def forward(self, skeleton_seq: torch.Tensor):
        return torch.stack(
            [self._align_single_sequence(seq) for seq in skeleton_seq],
            dim=0,
        )


class ValidCropResize(nn.Module):

    def __init__(self, p_interval, window, min_size):
        super().__init__()
        self.p_interval = p_interval
        self.window = window
        self.min_size = min_size

    def forward(self, data):
        M, T, V, C = data.shape
        begin = 0
        end = T
        valid_size = end - begin

        if len(self.p_interval) == 1:
            # deterministic center crop
            p = self.p_interval[0]
            bias = int((1 - p) * valid_size / 2)
            data = data[:, begin + bias : end - bias, :, :]
            cropped_length = data.shape[1]

        else:
            # random crop with interval
            p = float(np.random.rand(1) * (self.p_interval[1] - self.p_interval[0]) + self.p_interval[0])
            cropped_length = int(np.floor(valid_size * p))
            cropped_length = int(np.clip(cropped_length, self.min_size, valid_size))
            bias = np.random.randint(0, valid_size - cropped_length + 1)
            data = data[:, begin + bias : begin + bias + cropped_length, :, :]
            if data.shape[1] == 0:
                raise Exception("Invalid crop:", cropped_length, bias, valid_size)

        M, cropped_length, V, C = data.shape

        data = data.permute(3, 2, 0, 1).contiguous().view(C * V * M, cropped_length)
        data = data[None, None, :, :]

        data = F.interpolate(data, size=(C * V * M, self.window), mode="bilinear", align_corners=False).squeeze()
        data = data.contiguous().view(C, V, M, self.window).permute(2, 3, 1, 0).contiguous()
        return data
