import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import argparse
from nnunet_mednext import create_mednext_v1

# Argumentparser (hier kannst du zusätzlich die Ordner für Bilder und Masken übergeben)
parser = argparse.ArgumentParser()
parser.add_argument("-traindir", type=str, default="", help="training directory from syncache")
parser.add_argument("-valdir", type=str, default="", help="validation directory from syncache")
parser.add_argument("--images_dir", type=str, required=True, help="Pfad zum Ordner mit Trainingsbildern (.nii.gz)")
parser.add_argument("--masks_dir", type=str, required=True, help="Pfad zum Ordner mit Masken (.nii.gz)")
args = parser.parse_args()

traindir = args.traindir
valdir = args.valdir
images_dir = args.images_dir
masks_dir = args.masks_dir

def init_model(input_channels: int, num_classes: int, model_id: str, kernel_size: int, deep_supervision: bool):
    model = create_mednext_v1(
        num_input_channels=input_channels,
        num_classes=num_classes,
        model_id=model_id,
        kernel_size=kernel_size,
        deep_supervision=deep_supervision
    )
    return model

class NiftiDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        """
        images_dir: Pfad zum Ordner mit .nii.gz-Bildern
        masks_dir: Pfad zum Ordner mit .nii.gz-Masken
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
        self.mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.nii.gz')])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        

        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()


        image = (image - np.mean(image)) / np.std(image)


        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)


        if image.ndim == 3:
            image = image.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

def train_loop(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def main():

    input_channels = 1
    num_classes = 2
    model_id = 'S'
    kernel_size = 3
    deep_supervision = True
    num_epochs = 10
    batch_size = 1


    model = init_model(
        input_channels=input_channels,
        num_classes=num_classes,
        model_id=model_id,
        kernel_size=kernel_size,
        deep_supervision=deep_supervision
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = NiftiDataset(images_dir, masks_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    for epoch in range(num_epochs):
        loss = train_loop(model, train_dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
