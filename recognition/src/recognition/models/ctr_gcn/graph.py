from .tools import get_spatial_graph


class Graph:
    def __init__(self, layout="ntu-rgb+d", labeling_mode="spatial"):

        if layout == "ntu-rgb+d":
            self.num_node = 25
            self.self_link = [(i, i) for i in range(self.num_node)]
            self.inward_ori_index = [
                (1, 2),
                (2, 21),
                (3, 21),
                (4, 3),
                (5, 21),
                (6, 5),
                (7, 6),
                (8, 7),
                (9, 21),
                (10, 9),
                (11, 10),
                (12, 11),
                (13, 1),
                (14, 13),
                (15, 14),
                (16, 15),
                (17, 1),
                (18, 17),
                (19, 18),
                (20, 19),
                (22, 23),
                (23, 8),
                (24, 25),
                (25, 12),
            ]
            self.inward = [(i - 1, j - 1) for (i, j) in self.inward_ori_index]
            self.outward = [(j, i) for (i, j) in self.inward]
            self.neighbor = self.inward + self.outward
        elif layout == "coco17":
            self.num_node = 17
            self.self_link = [(i, i) for i in range(self.num_node)]
            self.inward_ori_index = [
                (15, 13),
                (13, 11),
                (16, 14),
                (14, 12),
                (11, 12),
                (5, 11),
                (6, 12),
                (5, 6),
                (5, 7),
                (6, 8),
                (7, 9),
                (8, 10),
                (1, 2),
                (0, 1),
                (0, 2),
                (1, 3),
                (2, 4),
                (3, 5),
                (4, 6),
            ]
            self.inward = [(i - 1, j - 1) for (i, j) in self.inward_ori_index]
            self.outward = [(j, i) for (i, j) in self.inward]
            self.neighbor = self.inward + self.outward
        elif layout == "coco33":
            self.num_node = 33
            self.self_link = [(i, i) for i in range(self.num_node)]
            self.inward_ori_index = [
                (16, 20),
                (16, 21),
                (16, 22),
                (15, 17),  # Left ankle to left big toe
                (15, 18),  # Left ankle to left small toe
                (15, 19),
                (15, 13),
                (13, 11),
                (16, 14),
                (14, 12),
                (11, 12),
                (5, 11),
                (6, 12),
                (5, 6),
                (5, 7),
                (6, 8),
                (7, 9),
                (8, 10),
                (1, 2),
                (0, 1),
                (0, 2),
                (1, 3),
                (2, 4),
                (3, 5),
                (4, 6),
                (9, 23),  # left wrist -> left thumb
                (9, 24),  # left wrist -> left index
                (9, 25),  # left wrist -> left middle
                (9, 26),  # left wrist -> left ring
                (9, 27),  # left wrist -> left pinky
                (10, 28),  # right wrist -> right thumb
                (10, 29),  # right wrist -> right index
                (10, 30),  # right wrist -> right middle
                (10, 31),  # right wrist -> right ring
                (10, 32),  # right wrist -> right pinky
            ]
            self.inward = [(i - 1, j - 1) for (i, j) in self.inward_ori_index]
            self.outward = [(j, i) for (i, j) in self.inward]
            self.neighbor = self.inward + self.outward
        else:
            raise ValueError("Unkown graph layout")
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == "spatial":
            A = get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A
