import os
import jittor as jt
import jittor.nn as nn
import numpy as np
from PIL import Image
import jittor.dataset as datasets


def load_annotations(ann_file):
    data_infos = {}
    with open(ann_file) as f:
        samples = [x.strip().split(' ') for x in f.readlines()]
        for i,j in samples:
            data_infos[i] = np.array((j), dtype=np.int64)
    return data_infos
class claDataset(datasets.ImageFolder):
    """Dataset for the finetuing stage."""
    def __init__(
        self,
        root,
        transform,
        pseudo_path
    ):
        super(claDataset, self).__init__(root, transform,pseudo_path)
        self.pseudo_path = pseudo_path
        self.img_label = load_annotations(self.pseudo_path)
        self.image_name = list(self.img_label.keys())
        self.label = list(self.img_label.values())
        self.data_dir = root
        self.image_path = [os.path.join(self.data_dir, img) for img in self.image_name]

    def __getitem__(self, index):
        """
        Returns:
        img (Tensor): The loaded image. (3 x H x W)
        pseudo (str): The generated pseudo label. (H x W)
        """

        path=self.image_path[index]
        img = Image.open(path).convert('RGB')
        pseudo_label=self.label[index]

        img = self.transform(img)
        return img, pseudo_label


