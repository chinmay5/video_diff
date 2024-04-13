import os

import torch
from omegaconf import OmegaConf

from environment_setup import PROJECT_ROOT_DIR
from inference_scripts.inference_utils import load_inference_unet_2, load_inference_unet_1, load_inference_unet_3
from utils.model_utils import create_model
from utils.train_utils import save_slices_as_images


def sample_volume(trainer, unet_nummer, config, starting_video):
    video_path = config.video_save_path
    os.makedirs(video_path, exist_ok=True)
    output_folder = os.path.join(video_path, f"Sampling", f"{unet_nummer=}")
    with torch.no_grad():
        # Sampling seems to be memory intensive.
        video = trainer.sample(stop_at_unet_number=unet_nummer, video_frames=config.num_frames,
                               start_image_or_video=starting_video)
        print(f"{video.shape=}")
        save_slices_as_images(video, output_folder=output_folder)


def setup(configs):
    device = torch.device(configs.device)
    # Instantiate the model
    imagen = create_model(config=configs, device=device)
    if configs.train_unet_number == 1:
        trainer1, starting_video = load_inference_unet_1(configs, device, imagen)
        sample_volume(trainer=trainer1, unet_nummer=1, config=configs, starting_video=starting_video)
    elif configs.train_unet_number == 2:
        trainer_2, starting_video = load_inference_unet_2(configs, device, imagen)
        sample_volume(trainer=trainer_2, unet_nummer=2, config=configs, starting_video=starting_video)
    elif configs.train_unet_number == 3:
        trainer_3, starting_video = load_inference_unet_3(configs, device, imagen)
        sample_volume(trainer=trainer_3, unet_nummer=3, config=configs, starting_video=starting_video)
    else:
        raise AttributeError(f"Invalid choice for {configs.train_unet_number=}")


if __name__ == "__main__":
    train_configs = OmegaConf.load(os.path.join(PROJECT_ROOT_DIR, 'configs/config.yaml'))
    model_config = OmegaConf.load(os.path.join(PROJECT_ROOT_DIR,'configs/models/model.yaml'))
    # From cli, do not give --server.port but, instead use server.port i.e. directly command line arguments.
    cli_conf = OmegaConf.from_cli()
    configs = OmegaConf.merge(train_configs, model_config, cli_conf)
    setup(configs=configs)
