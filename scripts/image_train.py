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
import os
import wandb

import torch

print("Is CUDA available:", torch.cuda.is_available())


# os.environ["WANDB_MODE"] = "disabled"

def yielder(loader):
    while True:
        yield from loader


def main():
    args = create_argparser().parse_args()
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    wandb.init(project="continual_diffusion", name=args.experiment_name, config=args, entity="generative_cl")
    os.environ["OPENAI_LOGDIR"] = f"results/{args.experiment_name}"

    dist_util.setup_dist(args)
    logger.configure()

    train_dataset, val_dataset, image_size, image_channels = base.__dict__[args.dataset](args.dataroot,
                                                                                         train_aug=args.train_aug)

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
    if not os.environ.get("WANDB_MODE") == "disabled":
        wandb.watch(model, log_freq=10)
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    train_dataset_splits, val_dataset_splits, task_output_space = data_split(dataset=train_dataset,
                                                                             return_classes=args.class_cond,
                                                                             return_task_as_class=args.use_task_index,
                                                                             dataset_name=args.dataset.lower(),
                                                                             num_batches=args.num_tasks,
                                                                             num_classes=n_classes,
                                                                             random_split=args.random_split,
                                                                             limit_data=args.limit_data,
                                                                             dirichlet_split_alpha=args.dirichlet,
                                                                             reverse=args.reverse,
                                                                             limit_classes=args.limit_classes)

    for task_id in range(args.num_tasks):
        train_dataset_loader = data.DataLoader(dataset=train_dataset_splits[task_id],
                                               batch_size=args.batch_size, shuffle=True,
                                               drop_last=True)

        if args.class_cond:
            if args.use_task_index:
                max_class = task_id
            else:
                raise NotImplementedError() #Classes seen so far for plotting and sampling
        else:
            max_class = None
        logger.log("training...")
        num_steps = args.num_steps
        if task_id == 0:
            num_steps = num_steps
        TrainLoop(
            model=model,
            diffusion=diffusion,
            task_id=task_id,
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
            num_steps=num_steps,
            image_size=args.image_size,
            in_channels=args.in_channels,
            class_cond=args.class_cond,
            max_class=max_class
        ).run_loop()


def create_argparser():
    defaults = dict(
        wandb_api_key="",
        experiment_name="test",
        dataroot="data/",
        dataset="MNIST",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=32,
        microbatch=-1,  # -1 disables microbatches
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
        scheduler_rate=1.0,
        use_task_index=True
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
