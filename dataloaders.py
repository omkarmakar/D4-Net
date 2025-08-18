import os
import glob
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

train_files = [y for x in os.walk('/kaggle/input/faceforensic/FaceForensic/FaceForensic/train') for y in glob.glob(os.path.join(x[0], '*.png'))]
val_files = [y for x in os.walk('/kaggle/input/faceforensic/FaceForensic/FaceForensic/validation') for y in glob.glob(os.path.join(x[0], '*.png'))]
test_files = [y for x in os.walk('/kaggle/input/faceforensic/FaceForensic/FaceForensic/test') for y in glob.glob(os.path.join(x[0], '*.png'))]
print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

def get_labels(paths):
    return [1 if 'fake' in path.lower() else 0 for path in paths]

train_labels = get_labels(train_files)
val_labels = get_labels(val_files)
test_labels = get_labels(test_files)

# ---------- Dataset Class ----------
class ImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = float(self.labels[idx])
        return img, torch.tensor(label, dtype=torch.float)

# ---------- Transforms ----------
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------- Data Loaders ----------
batch_size = 16
train_dataset = ImageDataset(train_files, train_labels, transform)
val_dataset = ImageDataset(val_files, val_labels, transform)
test_dataset = ImageDataset(test_files, test_labels, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)