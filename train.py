import torch
import torch.nn as nn
import torch.optim as optim

import os
from tqdm.auto import tqdm


def train_cycle_gan(dataloader, generator_A2B, generator_B2A,
                    discriminator_A, discriminator_B, num_epochs, root_dir,
                    prevEpoch=0, lambda_cycle=10.0):
    
    # Move models to GPU if CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator_A2B = generator_A2B.to(device)
    generator_B2A = generator_B2A.to(device)
    discriminator_A = discriminator_A.to(device)
    discriminator_B = discriminator_B.to(device)

    # Define loss criteria
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()

    # Define optimizers
    optimizer_G = optim.Adam(list(generator_A2B.parameters()) + list(generator_B2A.parameters()), lr=0.0002,
                             betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(discriminator_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(discriminator_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    checkpoint_interval = 5

    check_dir = os.path.join(root_dir, "checkpoints")
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)

    with open(os.path.join(root_dir, "losses.txt"), "a") as file:
        # Training loop
        for epoch in tqdm(range(num_epochs), desc="Epochs", leave=True, position=0):
            batch_progress = tqdm(enumerate(dataloader), desc="Batches", leave=False, position=1, total=len(dataloader))
            for i, batch in batch_progress:
                real_A = batch[0].to(device)
                real_B = batch[1].to(device)
                # Forward passes
                fake_B = generator_A2B(real_A)
                fake_A = generator_B2A(real_B)
                cycle_A = generator_B2A(fake_B)
                cycle_B = generator_A2B(fake_A)

                # Generator losses
                loss_GAN_A2B = criterion_GAN(discriminator_B(fake_B), torch.ones_like(discriminator_B(fake_B)).to(device))
                loss_cycle_A = criterion_cycle(cycle_A, real_A)

                loss_GAN_B2A = criterion_GAN(discriminator_A(fake_A), torch.ones_like(discriminator_A(fake_A)).to(device))
                loss_cycle_B = criterion_cycle(cycle_B, real_B)

                loss_G = loss_GAN_A2B + loss_GAN_B2A + lambda_cycle * (loss_cycle_A + loss_cycle_B)

                # Update generators
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

                # Discriminator losses
                loss_D_A = criterion_GAN(discriminator_A(real_A),
                                         torch.ones_like(discriminator_A(real_A)).to(device)) + criterion_GAN(
                    discriminator_A(fake_A.detach()), torch.zeros_like(discriminator_A(fake_A.detach())).to(device))
                loss_D_B = criterion_GAN(discriminator_B(real_B),
                                         torch.ones_like(discriminator_B(real_B)).to(device)) + criterion_GAN(
                    discriminator_B(fake_B.detach()), torch.zeros_like(discriminator_B(fake_B.detach())).to(device))

                # Update discriminators
                optimizer_D_A.zero_grad()
                loss_D_A.backward()
                optimizer_D_A.step()

                optimizer_D_B.zero_grad()
                loss_D_B.backward()
                optimizer_D_B.step()

            # Create checkpoints for the training in case of interruption
            if (epoch + 1) % checkpoint_interval == 0:
                torch.save(generator_A2B.state_dict(), os.path.join(check_dir, f'generator_A2B_epoch_{epoch + prevEpoch + 1}.pth'))
                torch.save(generator_B2A.state_dict(), os.path.join(check_dir, f'generator_B2A_epoch_{epoch + prevEpoch + 1}.pth'))
                torch.save(discriminator_A.state_dict(), os.path.join(check_dir, f'discriminator_A_epoch_{epoch + prevEpoch + 1}.pth'))
                torch.save(discriminator_B.state_dict(), os.path.join(check_dir, f'discriminator_B_epoch_{epoch + prevEpoch + 1}.pth'))

            loss_str = (f"\nEpoch {epoch + prevEpoch + 1}: "
                        f"Discriminator A Loss={loss_D_A.item():.4f},\t"
                        f"Discriminator B Loss={loss_D_B.item():.4f},\t"
                        f"Generator A2B Loss={loss_GAN_A2B.item():.4f},\t"
                        f"Generator B2A Loss={loss_GAN_B2A.item():.4f},\t"
                        f"Generator Cycle A Loss={loss_cycle_A.item():.4f},\t"
                        f"Generator Cycle B Loss={loss_cycle_B.item():.4f}")
            file.write(loss_str)
        file.close()

    return generator_A2B, generator_B2A