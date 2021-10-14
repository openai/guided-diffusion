# Evaluations

To compare different generative models, we use FID, sFID, Precision, Recall, and Inception Score. These metrics can all be calculated using batches of samples, which we store in `.npz` (numpy) files.

# Download batches

We provide pre-computed sample batches for the reference datasets, our diffusion models, and several baselines we compare against. These are all stored in `.npz` format.

Reference dataset batches contain pre-computed statistics over the whole dataset, as well as 10,000 images for computing Precision and Recall. All other batches contain 50,000 images which can be used to compute statistics and Precision/Recall.

Here are links to download all of the sample and reference batches:

 * LSUN
   * LSUN bedroom: [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/bedroom/VIRTUAL_lsun_bedroom256.npz)
     * [ADM (dropout)](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/bedroom/admnet_dropout_lsun_bedroom.npz)
     * [DDPM](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/bedroom/ddpm_lsun_bedroom.npz)
     * [IDDPM](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/bedroom/iddpm_lsun_bedroom.npz)
     * [StyleGAN](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/bedroom/stylegan_lsun_bedroom.npz)
   * LSUN cat: [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/cat/VIRTUAL_lsun_cat256.npz)
     * [ADM (dropout)](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/cat/admnet_dropout_lsun_cat.npz)
     * [StyleGAN2](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/cat/stylegan2_lsun_cat.npz)
   * LSUN horse: [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/horse/VIRTUAL_lsun_horse256.npz)
     * [ADM (dropout)](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/horse/admnet_dropout_lsun_horse.npz)
     * [ADM](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/horse/admnet_lsun_horse.npz)
   * ImageNet 64x64: [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/64/VIRTUAL_imagenet64_labeled.npz)
     * [ADM](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/64/admnet_imagenet64.npz)
     * [IDDPM](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/64/iddpm_imagenet64.npz)
     * [BigGAN](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/64/biggan_deep_imagenet64.npz)
   * ImageNet 128x128: [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/128/VIRTUAL_imagenet128_labeled.npz)
     * [ADM](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/128/admnet_imagenet128.npz)
     * [ADM-G](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/128/admnet_guided_imagenet128.npz)
     * [ADM-G, 25 steps](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/128/admnet_guided_25step_imagenet128.npz)
     * [BigGAN-deep (trunc=1.0)](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/128/biggan_deep_trunc1_imagenet128.npz)
   * ImageNet 256x256: [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz)
     * [ADM](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/admnet_imagenet256.npz)
     * [ADM-G](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/admnet_guided_imagenet256.npz)
     * [ADM-G, 25 step](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/admnet_guided_25step_imagenet256.npz)
     * [ADM-G + ADM-U](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/admnet_guided_upsampled_imagenet256.npz)
     * [ADM-U](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/admnet_upsampled_imagenet256.npz)
     * [BigGAN-deep (trunc=1.0)](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/biggan_deep_trunc1_imagenet256.npz)
   * ImageNet 512x512: [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz)
     * [ADM](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/admnet_imagenet512.npz)
     * [ADM-G](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/admnet_guided_imagenet512.npz)
     * [ADM-G, 25 step](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/admnet_guided_25step_imagenet512.npz)
     * [ADM-G + ADM-U](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/admnet_guided_upsampled_imagenet512.npz)
     * [ADM-U](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/admnet_upsampled_imagenet512.npz)
     * [BigGAN-deep (trunc=1.0)](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/biggan_deep_trunc1_imagenet512.npz)

# Run evaluations

First, generate or download a batch of samples and download the corresponding reference batch for the given dataset. For this example, we'll use ImageNet 256x256, so the refernce batch is `VIRTUAL_imagenet256_labeled.npz` and we can use the sample batch `admnet_guided_upsampled_imagenet256.npz`.

Next, run the `evaluator.py` script. The requirements of this script can be found in [requirements.txt](requirements.txt). Pass two arguments to the script: the reference batch and the sample batch. The script will download the InceptionV3 model used for evaluations into the current working directory (if it is not already present). This file is roughly 100MB.

The output of the script will look something like this, where the first `...` is a bunch of verbose TensorFlow logging:

```
$ python evaluator.py VIRTUAL_imagenet256_labeled.npz admnet_guided_upsampled_imagenet256.npz
...
computing reference batch activations...
computing/reading reference batch statistics...
computing sample batch activations...
computing/reading sample batch statistics...
Computing evaluations...
Inception Score: 215.8370361328125
FID: 3.9425574129223264
sFID: 6.140433703346162
Precision: 0.8265
Recall: 0.5309
```
