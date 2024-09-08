import data_loader as DL
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
import os


def initialize_datasets(root_dir, batch_size, percentage=0.8):
    # Create a dataset object with all images
    transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
    ])

    input_dir = os.path.join(root_dir, "NonRT")
    truth_dir = os.path.join(root_dir, "RT")

    dataset = DL.PortalDataset(input_dir, truth_dir, transform=transform)

    # Step 2: Calculate indices for training and testing subsets
    train_size = int(percentage * len(dataset))  # 80% for training
    test_size = len(dataset) - train_size
    train_indices, test_indices = random_split(range(len(dataset)), [train_size, test_size])

    if percentage == 0:
        test_indices = sorted(test_indices)

    # Use Subset to create training and testing datasets
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    # Create data loaders for training and testing datasets
    train_loader = None
    if percentage > 0:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    return train_loader, test_loader