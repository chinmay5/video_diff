# https://gist.github.com/HReynaud/2ac2c8a8342a3d9e2829c9113f4de274
import os
import time

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from utils.model_utils import create_model
from utils.train_utils import delay2str, save_slices_as_images, load_unet_1, load_unet_2, load_unet_3


def train_iterations(trainer, unet_nummer, config):
    # Trainning variables
    start_time = time.time()
    video_path = config.video_save_path
    checkpoint_path = config.checkpoint_path
    avg_loss = 1.0
    w_avg = 0.99
    target_loss = 0.005
    step_idx = unet_nummer - 1  # Iterations/ SGD steps for the specific u-net
    # The tasks that are supposed to be done once, should be done using accelerate
    # Perhaps, even the logging.
    if trainer.is_main:
        os.makedirs(video_path, exist_ok=True)
        print(f"Started training with target loss of {target_loss} or {config.max_iterations[step_idx]=}")
    trainer.accelerator.wait_for_everyone()
    while avg_loss > target_loss and trainer.steps[step_idx] < config.max_iterations[step_idx]:
        loss = trainer.train_step(unet_number=unet_nummer)
        avg_loss = w_avg * avg_loss + (1 - w_avg) * loss

        if trainer.is_main:
            # trainser.steps; step number for each of the u-nets
            print(
                f'Step: {trainer.steps[step_idx].item():<6} | Loss: {loss:<6.4f} Avg Loss: {avg_loss:<6.4f} |'
                f' {delay2str(time.time() - start_time):<10}',
                end='\r')  # type: ignore
            wandb.log({"train_loss": loss})
        # trainser.steps; step number for each of the u-nets
        # Sample only if we are loading the base model for later models in the cascade
        if (trainer.steps[step_idx] % config.sampling_steps[step_idx] == 0) and (
                step_idx == 0 or config.load_prev_checkpoint):
            trainer.accelerator.wait_for_everyone()
            trainer.accelerator.print()
            trainer.accelerator.print(f'Saving model and videos to wandb (it. {step_idx})')
            # Calculate validation loss
            valid_loss = np.mean([trainer.valid_step(unet_number=unet_nummer) for _ in range(10)])
            if trainer.is_main:
                # trainser.steps; step number for each of the u-nets
                print(
                    f'Step: {trainer.steps[step_idx].item():<6} | Loss: {loss:<6.4f} Avg Loss: {avg_loss:<6.4f} |'
                    f' {delay2str(time.time() - start_time):<10} | Valid Loss: {valid_loss:<8.4f}')  # type: ignore

                # Sample the video
                output_folder = os.path.join(video_path, f"Step_{trainer.steps[step_idx]}", f"{unet_nummer=}")
                with torch.no_grad():
                    video = trainer.sample(stop_at_unet_number=unet_nummer, video_frames=config.num_frames)
                    print(f"{video.shape=}")
                    wandb.log({"videos": wandb.Video((video.cpu().numpy() * 255).astype(np.uint8))})
                    save_slices_as_images(video, output_folder=output_folder)
        elif trainer.steps[step_idx] % config.sampling_steps[step_idx] == 0:
            # Not needed since imagen has complete integration with accelerator and does it automatically.
            # trainer.accelerator.wait_for_everyone()
            trainer.save(os.path.join(checkpoint_path, f"trained_video_{unet_nummer}.pt"))  # type: ignore
    # Final validation loss
    trainer.accelerator.wait_for_everyone()
    valid_loss = np.mean([trainer.valid_step(unet_number=unet_nummer) for _ in range(10)])
    if trainer.is_main:
        print(
            f'Step: {trainer.steps[step_idx].item():<6} | Loss: {loss:<6.4f} Avg Loss: {avg_loss:<6.4f} |'
            f' {delay2str(time.time() - start_time):<10} | Valid Loss: {valid_loss:<8.4f}')  # type: ignore


def main(configs):
    device = torch.device(configs.device)
    # Should be within the main thread as well
    # Create wandb session
    wandb.init(
        name=f"video_diff",
        project=configs.wandb.project,
        mode=configs.wandb.mode
    )
    # Error message ->'ImagenTrainer can only be initialized once per process - for the sake of distributed training,
    # you will now have to create a separate script to train each unet (or a script that accepts unet number as an
    # argument)'

    # Instantiate the model
    imagen = create_model(config=configs, device=device)
    if configs.train_unet_number == 1:
        trainer1 = load_unet_1(configs, device, imagen)
        # Train
        train_iterations(trainer=trainer1, unet_nummer=1, config=configs)
    elif configs.train_unet_number == 2:
        trainer_2 = load_unet_2(configs, device, imagen)
        train_iterations(trainer=trainer_2, unet_nummer=2, config=configs)
    elif configs.train_unet_number == 3:
        trainer_3 = load_unet_3(configs, device, imagen)
        train_iterations(trainer=trainer_3, unet_nummer=3, config=configs)
    else:
        raise AttributeError(f"Invalid choice for {configs.train_unet_number=}")


if __name__ == "__main__":
    # merging multiple configurations
    # conf = OmegaConf.merge(base_cfg, model_cfg, optimizer_cfg, dataset_cfg)
    model_config = OmegaConf.load('configs/models/model.yaml')
    print("LOADING CONFIGURATIONS FROM configs/distributed_config.yaml")
    train_configs = OmegaConf.load('configs/distributed_config.yaml')
    cli_conf = OmegaConf.from_cli()
    configs = OmegaConf.merge(train_configs, model_config, cli_conf)
    main(configs=configs)
