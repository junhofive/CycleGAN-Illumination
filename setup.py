import os
import torch

import util
import cycle_gan as CG
import train
import test


def experiment_CycleGAN(output_dir,
                        epochs, 
                        saved_location=None, 
                        trained_epoch=0,
                        dataType='Portal'):
    
    train_loader, test_loader = util.initialize_datasets(dataType, percentage=0.8)

    generator_A2B = CG.UNet(3, 3)
    generator_B2A = CG.UNet(3, 3)
    discriminator_A = CG.PatchGANDiscriminator()
    discriminator_B = CG.PatchGANDiscriminator()

    if saved_location:
        trained_dir = saved_location
        generator_A2B.load_state_dict(torch.load(os.path.join(trained_dir, f"generator_A2B_epoch_{trained_epoch}.pth")), strict=False)
    else:
        generator_A2B, generator_B2A = train.train_cycle_gan(train_loader,
                                               generator_A2B,
                                               generator_B2A,
                                               discriminator_A,
                                               discriminator_B,
                                               num_epochs=epochs,
                                               prevEpoch=trained_epoch, root_dir=output_dir)

    test_dir = f'{dataType}_Validation'
    train_loader, test_loader = util.initialize_datasets(dataType, percentage=0.8)
    test.evaluate_cycleGAN(generator_A2B, test_loader, output_dir,
                      f"CycleGAN_{dataType}_{trained_epoch}_metrics.txt", trained_epoch=(trained_epoch + epochs))