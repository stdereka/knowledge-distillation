"""
This module contains classes for datasets compatible with Pytorch
"""
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, files: list, label_encoder: LabelEncoder, teacher_labels=None, size=128, augs=False):
        """
        :param files: list of files to include in dataset
        :param label_encoder: sklearn label encoder for mapping class labels
        :param teacher_labels: path to dark knowledge of teacher model
        :param size: size of a picture in a dataset
        :param augs: use augmentations
        """
        self.files = files

        # Class label is a parent directory
        self.labels = [path.parent.name for path in self.files]
        self.labels = label_encoder.transform(self.labels)

        if teacher_labels:
            self.teacher_labels = np.load(teacher_labels)
        else:
            self.teacher_labels = []

        self.transformations = transforms.Compose([
                                                   transforms.Resize((size, size)),
                                                   transforms.ToTensor(),
                                                   ])

        if augs:
            self.augs = transforms.Compose([
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.RandomAffine(degrees=(-30, 30), scale=(0.75, 1.5))
                ], p=0.7),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomPerspective(p=0.5)
            ])
        else:
            self.augs = None

    def __getitem__(self, index):
        path = self.files[index]
        label = self.labels[index]

        image = Image.open(path).convert('RGB')
        image.load()

        if self.augs:
            image = self.augs(image)

        image = self.transformations(image)

        if len(self.teacher_labels) > 0:
            teacher_label = self.teacher_labels[index]
            return image, (label, teacher_label)
        else:
            # Returning in this format allows to use the same code for training with and without teacher model
            return image, (label, )

    def __len__(self):
        return len(self.files)
