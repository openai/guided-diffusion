"""
Approximate the bits/dimension for an image model.
"""

import argparse
import os

import numpy as np
import torch.distributed as dist
from dataloaders import base

from dataloaders.datasetGen import data_split
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torch
# from torch.utils import data
# from validation import Validator

def yielder(loader):
    while True:
        yield from loader

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    os.environ["OPENAI_LOGDIR"] = f"results/{args.experiment_name}"

    if args.seed:
        print("Using manual seed = {}".format(args.seed))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("WARNING: Not using manual seed - your experiments will not be reproducible")

    # os.environ["WANDB_API_KEY"] = args.wandb_api_key
    # wandb.init(project="continual_diffusion", name=args.experiment_name, config=args, entity="generative_cl")
    # os.environ["OPENAI_LOGDIR"] = f"results/{args.experiment_name}"

    logger.configure()

    train_dataset, val_dataset, image_size, image_channels = base.__dict__[args.dataset](args.dataroot,
                                                                                         train_aug=args.train_aug)

    args.image_size = image_size
    args.in_channels = image_channels
    if args.dataset.lower() == "celeba":
        n_classes = 10
    else:
        n_classes = train_dataset.number_classes

    args.num_classes = args.num_tasks  # n_classes
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model_path = f"results/{args.experiment_name}/{args.model_path}"
    model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )
    # if not os.environ.get("WANDB_MODE") == "disabled":
    #     wandb.watch(model, log_freq=10)
    model.to(dist_util.dev())

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

    val_loaders = []
    for task_id in range(args.num_tasks):
        val_data = val_dataset_splits[task_id]
        val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
        val_loaders.append(val_loader)


    logger.log("evaluating...")
    diffusion.dae_model = False
    for task_id in range(args.num_tasks):
        data = yielder(val_loaders[task_id])
        run_bpd_evaluation(model, diffusion, data, args.num_samples, args.clip_denoised)


def run_bpd_evaluation(model, diffusion, data, num_samples, clip_denoised):
    all_bpd = []
    all_metrics = {"vb": [], "mse": [], "xstart_mse": []}
    num_complete = 0
    while num_complete < num_samples:
        batch, model_kwargs = next(data)
        batch = batch.to(dist_util.dev())
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        minibatch_metrics = diffusion.calc_bpd_loop(
            model, batch, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )

        for key, term_list in all_metrics.items():
            terms = minibatch_metrics[key].mean(dim=0) / dist.get_world_size()
            dist.all_reduce(terms)
            term_list.append(terms.detach().cpu().numpy())

        total_bpd = minibatch_metrics["total_bpd"]
        total_bpd = total_bpd.mean() / dist.get_world_size()
        dist.all_reduce(total_bpd)
        all_bpd.append(total_bpd.item())
        num_complete += dist.get_world_size() * batch.shape[0]

        logger.log(f"done {num_complete} samples: bpd={np.mean(all_bpd)}")

    if dist.get_rank() == 0:
        for name, terms in all_metrics.items():
            out_path = os.path.join(logger.get_dir(), f"{name}_terms.npz")
            logger.log(f"saving {name} terms to {out_path}")
            np.savez(out_path, np.mean(np.stack(terms), axis=0))

    dist.barrier()
    logger.log("evaluation complete")


def create_argparser():
    defaults = dict(
        seed=13,
        wandb_api_key="",
        experiment_name="test",
        dataroot="data/",
        dataset="MNIST",
        data_dir="",
        clip_denoised=True,
        num_samples=1000,
        batch_size=1,
        model_path="",
        gpu_id=-1,
        reverse=False,
        dirichlet=None,
        num_tasks=5,
        limit_classes=-1,
        random_split=False,
        train_aug=False,
        limit_data=None,
        use_task_index=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
