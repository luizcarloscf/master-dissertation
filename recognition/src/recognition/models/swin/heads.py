import torch.nn as nn
import torch

RIGHT_HAND_JOINTS = [6, 8, 10, 28, 29, 30, 31, 32]
LEFT_HAND_JOINTS = [5, 7, 9, 23, 24, 25, 26, 27]
BODY_JOINTS = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]


class ActionHeadLinprobe(nn.Module):
    def __init__(self, dim_feat=512, num_classes=60, num_joints=25):
        super().__init__()
        self.fc = nn.Linear(dim_feat, num_classes)

    def forward(self, feat):
        N, M, T, V, C = feat.shape
        feat = feat.mean(dim=[1, 2, 3])
        feat = self.fc(feat)
        return feat


class ActionHeadFinetune(nn.Module):
    def __init__(self, dropout_ratio=0.0, dim_feat=512, num_classes=60, num_joints=25, hidden_dim=2048):
        super(ActionHeadFinetune, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_feat * num_joints, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, feat):
        """
        Input: (N, M, T, J, C)
        """
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 3, 4, 2)  # (N, M, T, J, C) -> (N, M, J, C, T)
        feat = feat.mean(dim=-1)
        feat = feat.reshape(N, M, -1)  # (N, M, J*C)
        feat = feat.mean(dim=1)
        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)
        feat = self.fc2(feat)
        return feat
