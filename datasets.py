from torch.utils.data.dataset import Dataset
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder


class Imagewoof(Dataset):
    def __init__(self, files: list, label_encoder: LabelEncoder):
        self.files = files

        self.labels = [path.parent.name for path in self.files]
        self.labels = label_encoder.transform(self.labels)

        self.transformations = transforms.Compose([
                                                   transforms.Resize((256, 256)),
                                                   transforms.ToTensor(),
                                                   ])

    def __getitem__(self, index):
        path = self.files[index]
        label = self.labels[index]

        image = Image.open(path).convert('RGB')
        image.load()

        image = self.transformations(image)

        return image, label

    def __len__(self):
        return len(self.files)
