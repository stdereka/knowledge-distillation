from torch.utils.data.dataset import Dataset
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import numpy as np


class Imagewoof(Dataset):
    def __init__(self, files: list, label_encoder: LabelEncoder, teacher_labels=None, size=256, augs=False):
        self.files = files

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
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.RandomAffine(degrees=(-30, 30), scale=(0.75, 1.25))
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
            return image, (label, )

    def __len__(self):
        return len(self.files)
