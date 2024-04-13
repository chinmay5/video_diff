from imagen_pytorch import Unet3D, ElucidatedImagen, NullUnet


def create_model(config, device):
    # Define model
    # 64 used by multiple models. So, should be fine.
    # Although, it raises a warning but this warning is perhaps more tailored to images.
    # https://github.com/lucidrains/imagen-pytorch/issues/58
    dim_mults_0, dim_mults_1, dim_mults_2 = config.model.dim_mults[0], config.model.dim_mults[1], config.model.dim_mults[2]
    unet1 = Unet3D(dim=config.model.dims[0], channels=config.model.channels[0],
                   dim_mults=(dim_mults_0[0], dim_mults_0[1], dim_mults_0[2], dim_mults_0[3])).to(device)
    unet2 = Unet3D(dim=config.model.dims[1], channels=config.model.channels[1],
                   dim_mults=(dim_mults_1[0], dim_mults_1[1], dim_mults_1[2], dim_mults_1[3]), lowres_cond=True).to(
        device)
    # https://github.com/lucidrains/imagen-pytorch/issues/300
    unet3 = Unet3D(dim=config.model.dims[2], channels=config.model.channels[2],
                   dim_mults=(dim_mults_2[0], dim_mults_2[1], dim_mults_2[2], dim_mults_2[3]),
                   lowres_cond=True, layer_attns=False).to(device)
    # elucidated imagen, which contains the unets above (base unet and super resoluting ones)
    model = ElucidatedImagen(
        unets=(unet1, unet2, unet3),
        image_sizes=(config.model.image_sizes[0], config.model.image_sizes[1], config.model.image_sizes[2]),
        channels=config.model.channels[0],
        random_crop_sizes=(config.model.crop_sizes[0], config.model.crop_sizes[1], config.model.crop_sizes[2]),
        temporal_downsample_factor=(
        config.model.temporal_downscale[0], config.model.temporal_downscale[1], config.model.temporal_downscale[2]),
        # in this example, the first unet would receive the video temporally downsampled by 2x
        # num_sample_steps=10,
        cond_drop_prob=0.1,
        sigma_min=0.002,  # min noise level
        sigma_max=(80, 80, 80),
        sigma_data=0.5,  # standard deviation of data distribution
        rho=7,  # controls the sampling schedule
        P_mean=-1.2,  # mean of log-normal distribution from which noise is drawn for training
        P_std=1.2,  # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn=80,  # parameters for stochastic sampling - depends on dataset, Table 5 in apper
        S_tmin=0.05,
        S_tmax=50,
        S_noise=1.003,
        condition_on_text=config.model.condition_on_text
    )
    return model

