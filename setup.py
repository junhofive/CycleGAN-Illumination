import os
import torch

import util
import cycle_gan as CG
import train
import test


def experiment_CycleGAN(data_dir, save_dir,
                        dataset, batch_size=1,
                        epochs=100, groups=1,
                        saved_location=None, trained_epoch=0,
                        retrain=False, root_dir=None, old_model=False, dataType='Portal'):
    if root_dir == None:
        print("No root directory")
        return
    input_dir = os.path.join(root_dir, data_dir)

    train_loader, test_loader = util.initialize_datasets(input_dir, batch_size, percentage=0.8)

    generator_A2B = CG.UNet(3, 3)
    generator_B2A = CG.UNet(3, 3)
    discriminator_A = CG.PatchGANDiscriminator()
    discriminator_B = CG.PatchGANDiscriminator()

    new_dir = 'W:\\Research'
    out_dir = os.path.join(new_dir, save_dir)

    if saved_location:
        trained_dir = os.path.join(new_dir, saved_location)
        trained_dir = os.path.join(trained_dir, 'checkpoints')
        if old_model:
            generator_A2B.load_state_dict(
                torch.load(os.path.join(trained_dir, f"generator_A2B_epoch_{trained_epoch}.pth")), strict=False)
        else:
            generator_A2B.load_state_dict(torch.load(os.path.join(trained_dir, f"generator_A2B_epoch_{trained_epoch}.pth")))
        if retrain == True:
            train_loader, test_loader = util.initialize_datasets(input_dir, dataset, batch_size)
            generator_A2B, generator_B2A = train.train_cycle_gan(train_loader,
                                                       generator_A2B,
                                                       generator_B2A,
                                                       discriminator_A,
                                                       discriminator_B,
                                                       num_epochs=epochs,
                                                       prevEpoch=trained_epoch, root_dir=out_dir)
    else:
        # train_loader, test_loader = initialize_datasets(input_dir, dataset, batch_size)
        generator_A2B, generator_B2A = train.train_cycle_gan(train_loader,
                                               generator_A2B,
                                               generator_B2A,
                                               discriminator_A,
                                               discriminator_B,
                                               num_epochs=epochs,
                                               prevEpoch=trained_epoch, root_dir=out_dir)

    input_dir = os.path.join(root_dir, 'PortalValidationSet256')
    train_loader, test_loader = util.initialize_datasets(input_dir, dataset, batch_size, percentage=0.8)
    test.evaluate_cycleGAN(generator_A2B, test_loader, out_dir,
                      f"CycleGAN_{dataType}_{trained_epoch}_metrics.txt", trained_epoch=(trained_epoch + epochs))