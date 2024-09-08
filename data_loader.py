import os
import re
import random
from PIL import Image
from torch.utils.data import Dataset


class PortalDataset(Dataset):
    def __init__(self, inputDir, targetDir, transform):
        self.inputDir = inputDir
        self.targetDir = targetDir

        self.transform = transform

        all_inputs = os.listdir(inputDir)
        all_targets = os.listdir(targetDir)

        self.input_list = sorted(all_inputs, key=self.natural_keys)
        self.target_list = sorted(all_targets, key=self.natural_keys)

        random.shuffle(self.input_list)
        random.shuffle(self.target_list)


    def __len__(self):
        return max(len(self.input_list), len(self.target_list))

    def __getitem__(self, idx):
        input_name = self.input_list[idx]
        target_name = self.target_list[idx]

        input_path = os.path.join(self.inputDir, input_name)
        target_path = os.path.join(self.targetDir, target_name)

        input_image = self.load_image(input_path)
        input_tensor = self.transform(input_image)

        target_image = self.load_image(target_path)
        target_tensor = self.transform(target_image)

        return input_tensor, target_tensor

    def load_image(self, filepath):
        image = Image.open(filepath).convert('RGB')
        return image

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]