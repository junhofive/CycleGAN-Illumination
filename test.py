import os
import time

import torch
from torchvision import transforms
from tqdm.auto import tqdm
from PIL import Image, ImageDraw, ImageFont
import lpips

import metrics



def evaluate_cycleGAN(generator_A2B, dataloader, output_dir, results_file, trained_epoch=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator_A2B = generator_A2B.to(device).eval()


    out_dir = os.path.join(output_dir, f'validation_epoch{trained_epoch}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        count = 1
        while (os.path.exists(out_dir)):
            out_dir = os.path.join(output_dir, f'validation_epoch{trained_epoch}_{count}')
            count += 1
        os.makedirs(out_dir)

    total_samples = 0
    loss_fn = lpips.LPIPS(net='alex')

    with open(os.path.join(out_dir, results_file), 'w') as f:
        with torch.no_grad():
            batch_progress = tqdm(enumerate(dataloader), desc="Batches", leave=False, position=0, total=len(dataloader))
            for _, batch in batch_progress:
                # Unpack the batch
                input_image, target_image = batch

                # Move the images to the device
                input_image = input_image.to(device)
                target_image = target_image.to(device)

                # Generate the output image
                start_time = time.time()
                output_image = generator_A2B(input_image)
                end_time = time.time()
                execution_time = end_time - start_time

                output_image_tensor = output_image.cpu()
                target_image_tensor = target_image.cpu()

                # Move the images back to the CPU and convert to numpy for evaluation
                input_image_np = input_image.cpu().numpy().transpose(0, 2, 3, 1)
                output_image_np = output_image.cpu().numpy().transpose(0, 2, 3, 1)
                target_image_np = target_image.cpu().numpy().transpose(0, 2, 3, 1)

                # Loop through each image in the batch
                for i in range(input_image.size(0)):
                    # Save side by side
                    mets = metrics.calculate_metrics(target_image_np[i], output_image_np[i], target_image_tensor[i],
                                                     output_image_tensor[i], loss_fn)
                    save_side_by_side(input_image[i].cpu(), output_image[i].cpu(),
                                      target_image[i].cpu(),
                                      os.path.join(out_dir, f"output_{total_samples + i}.png"), mets)

                    # Log to file
                    f.write(f"Image {total_samples + i}: Time = {execution_time}, " +
                            ', '.join([f'{k} = {v}' for k, v in mets.items()]) + '\n')
                total_samples += input_image.size(0)


def save_side_by_side(input_image, output_image, target_image, path, metrics=None):
    # Convert PyTorch tensors to PIL images
    input_image = transforms.ToPILImage()(input_image)
    output_image = transforms.ToPILImage()(output_image)
    target_image = transforms.ToPILImage()(target_image)

    # Create a blank image with white background
    width, height = input_image.size
    new_image = Image.new("RGB", (3 * width, height + 50), (255, 255, 255))
    new_image.paste(input_image, (0, 0))
    new_image.paste(output_image, (width, 0))
    new_image.paste(target_image, (2 * width, 0))

    # Draw metrics
    if metrics:
        draw = ImageDraw.Draw(new_image)
        font = ImageFont.load_default()
        text = ', '.join([f'{key.upper()}: {value:.6f}' for key, value in metrics.items()])
        draw.text((10, height + 10), text, font=font, fill=(0, 0, 0))

    new_image.save(path)