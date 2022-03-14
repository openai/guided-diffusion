"""
Train a diffusion model on images.
"""

import argparse

from dataloaders import base
from dataloaders.datasetGen import *

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from torch.utils import data
from guided_diffusion.train_util import TrainLoop


def yielder(loader):
    while True:
        yield from loader

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args)
    logger.configure()

    train_dataset, val_dataset, image_size, image_channels = base.__dict__[args.dataset](args.dataroot, args.train_aug)

    args.image_size = image_size
    args.in_channels = image_channels
    if args.dataset.lower() == "celeba":
        n_classes = 10
    else:
        n_classes = train_dataset.number_classes

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    train_dataset_splits, val_dataset_splits, task_output_space = data_split(dataset=train_dataset,
                                                                             dataset_name=args.dataset.lower(),
                                                                             num_batches=args.num_tasks,
                                                                             num_classes=n_classes,
                                                                             random_split=args.random_split,
                                                                             limit_data=args.limit_data,
                                                                             dirichlet_split_alpha=args.dirichlet,
                                                                             reverse=args.reverse,
                                                                             limit_classes=args.limit_classes)

    for task_id in range(args.num_tasks):
        train_dataset_loader = data.DataLoader(dataset=train_dataset_splits[0],
                                               batch_size=args.batch_size, shuffle=True,
                                               drop_last=True)


        logger.log("training...")
        TrainLoop(
            model=model,
            diffusion=diffusion,
            task_id = task_id,
            data=yielder(train_dataset_loader),
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            scheduler_rate=args.scheduler_rate,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            plot_interval=args.plot_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            num_steps=args.num_steps,
            image_size=args.image_size,
            in_channels=args.in_channels
        ).run_loop()


def create_argparser():
    defaults = dict(
        dataroot="data/",
        dataset="MNIST",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=32,
        microbatch =-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=5000,
        plot_interval=1000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_id=-1,
        reverse=False,
        dirichlet=None,
        num_tasks=5,
        limit_classes=-1,
        random_split=False,
        train_aug=False,
        limit_data=None,
        num_steps=10000,
        scheduler_rate=1
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
