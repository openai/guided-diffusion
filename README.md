# guided-diffusion

This is the codebase for [Diffusion Models Beat GANS on Image Synthesis](http://arxiv.org/abs/2105.05233).

This repository is based on [openai/improved-diffusion](https://github.com/openai/improved-diffusion), with modifications for classifier conditioning and architecture improvements.

# Usage

Training diffusion models is described in the [parent repository](https://github.com/openai/improved-diffusion). Training a classifier is similar. We assume you have put training hyperparameters into a `TRAIN_FLAGS` variable, and classifier hyperparameters into a `CLASSIFIER_FLAGS` variable. Then you can run:

```
mpiexec -n N python scripts/classifier_train.py --data_dir path/to/imagenet $TRAIN_FLAGS $CLASSIFIER_FLAGS
```

Make sure to divide the batch size in `TRAIN_FLAGS` by the number of MPI processes you are using.

Here are flags for training the 128x128 classifier. You can modify these for training classifiers at other resolutions:

```sh
TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 256 --lr 3e-4 --save_interval 10000 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
```

For sampling from a 128x128 classifier-guided model, 25 step DDIM:

```sh
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --image_size 128 --learn_sigma True --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_scale 1.0 --classifier_use_fp16 True"
SAMPLE_FLAGS="--batch_size 4 --num_samples 50000 --timestep_respacing ddim25 --use_ddim True"
mpiexec -n N python scripts/classifier_sample.py \
    --model_path /path/to/model.pt \
    --classifier_path path/to/classifier.pt \
    $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS
```

To sample for 250 timesteps without DDIM, replace `--timestep_respacing ddim25` to `--timestep_respacing 250`, and replace `--use_ddim True` with `--use_ddim False`.
