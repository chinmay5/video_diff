import os

import numpy as np
from imagen_pytorch import ImagenTrainer

from dataset.cow_mra import TopCowMraDataset
import cv2


def load_unet_3(configs, device, imagen):
    # Now we can train the second u-net. The second u-net would be trained next.]
    # We first load the state
    # https://github.com/lucidrains/imagen-pytorch/issues/241
    trainer = ImagenTrainer(imagen=imagen).to(device)
    if configs.load_prev_checkpoint:
        assert os.path.exists(configs.unet_2_checkpoint), "No checkpoint for Unet 2"
        print("Loading Unet temporal super-res checkpoint from {}".format(configs.unet_2_checkpoint))
        trainer.load(configs.unet_2_checkpoint)
    if configs.resume:
        assert os.path.exists(configs.unet_3_checkpoint), "No checkpoint for Unet 3"
        print("Loading Unet spatial super-res checkpoint from {}".format(configs.unet_3_checkpoint))
        trainer.load(configs.unet_3_checkpoint)
    trainer.add_train_dataset(TopCowMraDataset(data_dir=configs.data_dir, split='train'),
                              batch_size=configs.batch_size, num_workers=configs.num_workers)
    trainer.add_valid_dataset(TopCowMraDataset(data_dir=configs.data_dir, split='val'),
                              batch_size=configs.batch_size, num_workers=configs.num_workers)
    return trainer


def load_unet_2(configs, device, imagen):
    # Now we can train the second u-net. The second u-net would be trained next.]
    # We first load the state
    # https://github.com/lucidrains/imagen-pytorch/issues/241
    trainer = ImagenTrainer(imagen=imagen).to(device)
    if configs.load_prev_checkpoint:
        assert os.path.exists(configs.unet_1_checkpoint), "No checkpoint for Unet 1"
        print("Loading Unet base checkpoint from {}".format(configs.unet_1_checkpoint))
        trainer.load(configs.unet_1_checkpoint)
    if configs.resume:
        assert os.path.exists(configs.unet_2_checkpoint), "No checkpoint for Unet 2"
        print("Loading Unet base checkpoint from {}".format(configs.unet_2_checkpoint))
        trainer.load(configs.unet_2_checkpoint)
    trainer.add_train_dataset(TopCowMraDataset(data_dir=configs.data_dir, split='train'),
                              batch_size=configs.batch_size, num_workers=configs.num_workers)
    trainer.add_valid_dataset(TopCowMraDataset(data_dir=configs.data_dir, split='val'),
                              batch_size=configs.batch_size, num_workers=configs.num_workers)
    return trainer


def load_unet_1(configs, device, imagen):
    trainer = ImagenTrainer(
        imagen=imagen,
        lr=configs.lr
    ).to(device)
    # If you want to resume training from a checkpoint
    if configs.resume:
        assert os.path.exists(configs.unet_1_checkpoint), "No checkpoint for Unet 1. Can not resume"
        print("Loading Unet base checkpoint from {}".format(configs.unet_1_checkpoint))
        trainer.load(configs.unet_1_checkpoint)
    trainer.add_train_dataset(TopCowMraDataset(data_dir=configs.data_dir, split='train'),
                              batch_size=configs.batch_size, num_workers=configs.num_workers)
    trainer.add_valid_dataset(TopCowMraDataset(data_dir=configs.data_dir, split='val'),
                              batch_size=configs.batch_size, num_workers=configs.num_workers)
    return trainer


def save_slices_as_images(volume, output_folder):
    # Iterate over each slice along the temporal dimension
    os.makedirs(output_folder, exist_ok=True)
    volume = volume.squeeze().cpu().numpy()
    for i in range(volume.shape[0]):
        # Extract the slice
        slice_image = volume[i, :, :]
        # Scaling the values in the right range. Not doing so would lead to just black color
        slice_image = (slice_image * 255).astype(np.uint8)
        # Save the slice as an image
        cv2.imwrite(f"{output_folder}/slice_{i}.png", slice_image)


def create_video_from_slices(volume, output_video_path, fps=10):
    # Initialize video writer
    height, width = volume.shape[1], volume.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change codec accordingly
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Iterate over each slice along the temporal dimension
    for i in range(volume.shape[0]):
        # Extract the slice
        slice_image = volume[i, :, :]

        # Convert slice to BGR format (OpenCV uses BGR by default)
        slice_image_bgr = cv2.cvtColor(slice_image, cv2.COLOR_GRAY2BGR)

        # Write the slice to the video
        video_writer.write(slice_image_bgr)

    # Release the video writer
    video_writer.release()


def delay2str(t):
    t = int(t)
    secs = t % 60
    mins = (t // 60) % 60
    hours = (t // 3600) % 24
    days = t // 86400
    string = f"{secs}s"
    if mins:
        string = f"{mins}m {string}"
    if hours:
        string = f"{hours}h {string}"
    if days:
        string = f"{days}d {string}"
    return string
