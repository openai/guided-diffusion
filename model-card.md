# Overview

These are diffusion models and noised image classifiers described in the paper [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233).
Included in this release are the following models:

 * Noisy ImageNet classifiers at resolutions 64x64, 128x128, 256x256, 512x512
 * A class-unconditional ImageNet diffusion model at resolution 256x256
 * Class conditional ImageNet diffusion models at 64x64, 128x128, 256x256, 512x512 resolutions
 * Class-conditional ImageNet upsampling diffusion models: 64x64->256x256, 128x128->512x512
 * Diffusion models trained on three LSUN classes at 256x256 resolution: cat, horse, bedroom

# Datasets

All of the models we are releasing were either trained on the [ILSVRC 2012 subset of ImageNet](http://www.image-net.org/challenges/LSVRC/2012/) or on single classes of [LSUN](https://arxiv.org/abs/1506.03365).
Here, we describe characteristics of these datasets which impact model behavior:

**LSUN**: This dataset was collected in 2015 using a combination of human labeling (from Amazon Mechanical Turk) and automated data labeling.
 * Each of the three classes we consider contain over a million images.
 * The dataset creators found that the label accuracy was roughly 90% across the entire LSUN dataset when measured by trained experts.
 * Images are scraped from the internet, and LSUN cat images in particular tend to often follow a “meme” format.
 * We found that there are occasionally humans in these photos, including faces, especially within the cat class.  

**ILSVRC 2012 subset of ImageNet**: This dataset was curated in 2012 and consists of roughly one million images, each belonging to one of 1000 classes.
 * A large portion of the classes in this dataset are animals, plants, and other naturally-occurring objects.
 * Many images contain humans, although usually these humans aren’t reflected by the class label (e.g. the class “Tench, tinca tinca” contains many photos of people holding fish).

# Performance

These models are intended to generate samples consistent with their training distributions.
This has been measured in terms of FID, Precision, and Recall.
These metrics all rely on the representations of a [pre-trained Inception-V3 model](https://arxiv.org/abs/1512.00567),
which was trained on ImageNet, and so is likely to focus more on the ImageNet classes (such as animals) than on other visual features (such as human faces).

Qualitatively, the samples produced by these models often look highly realistic, especially when a diffusion model is combined with a noisy classifier.

# Intended Use

These models are intended to be used for research purposes only.
In particular, they can be used as a baseline for generative modeling research, or as a starting point to build off of for such research.

These models are not intended to be commercially deployed.
Additionally, they are not intended to be used to create propaganda or offensive imagery.

Before releasing these models, we probed their ability to ease the creation of targeted imagery, since doing so could be potentially harmful.
We did this either by fine-tuning our ImageNet models on a target LSUN class, or through classifier guidance with publicly available [CLIP models](https://github.com/openai/CLIP).
 * To probe fine-tuning capabilities, we restricted our compute budget to roughly $100 and tried both standard fine-tuning,
and a diffusion-specific approach where we train a specialized classifier for the LSUN class. The resulting FIDs were significantly worse than publicly available GAN models, indicating that fine-tuning an ImageNet diffusion model does not significantly lower the cost of image generation.
 * To probe guidance with CLIP, we tried two approaches for using pre-trained CLIP models for classifier guidance. Either we fed the noised image to CLIP directly and used its gradients, or we fed the diffusion model's denoised prediction to the CLIP model and differentiated through the whole process. In both cases, we found that it was difficult to recover information from the CLIP model, indicating that these diffusion models are unlikely to make it significantly easier to extract knowledge from CLIP compared to existing GAN models.

# Limitations

These models sometimes produce highly unrealistic outputs, particularly when generating images containing human faces.
This may stem from ImageNet's emphasis on non-human objects.

While classifier guidance can improve sample quality, it reduces diversity, resulting in some modes of the data distribution being underrepresented.
This can potentially amplify existing biases in the training dataset such as gender and racial biases.

Because ImageNet and LSUN contain images from the internet, they include photos of real people, and the model may have memorized some of the information contained in these photos.
However, these images are already publicly available, and existing generative models trained on ImageNet have not demonstrated significant leakage of this information.