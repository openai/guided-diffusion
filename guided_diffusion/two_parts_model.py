import torch as th
import torch.nn as nn


from guided_diffusion.unet import UNetModel


class TwoPartsUNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            model_switching_timestep=None
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        self.switching_point = model_switching_timestep

        self.unet_1 = UNetModel(image_size,
                                in_channels,
                                model_channels,
                                out_channels,
                                num_res_blocks,
                                attention_resolutions,
                                dropout,
                                channel_mult,
                                conv_resample,
                                dims,
                                num_classes,
                                use_checkpoint,
                                use_fp16,
                                num_heads,
                                num_head_channels,
                                num_heads_upsample,
                                use_scale_shift_norm,
                                resblock_updown,
                                use_new_attention_order)

        self.unet_2 = UNetModel(image_size,
                                in_channels,
                                model_channels,
                                out_channels,
                                num_res_blocks,
                                attention_resolutions,
                                dropout,
                                channel_mult,
                                conv_resample,
                                dims,
                                num_classes,
                                use_checkpoint,
                                use_fp16,
                                num_heads,
                                num_head_channels,
                                num_heads_upsample,
                                use_scale_shift_norm,
                                resblock_updown,
                                use_new_attention_order)

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        timesteps_unet_1 = timesteps < self.switching_point
        timesteps_unet_2 = ~timesteps_unet_1
        out = th.zeros(x.shape[0],x.shape[1]*2,x.shape[2],x.shape[3],device=x.device)
        if timesteps_unet_1.sum()>0:
            x_1 = self.unet_1(x[timesteps_unet_1], timesteps[timesteps_unet_1], y[timesteps_unet_1])
            out[timesteps_unet_1] = x_1
        if timesteps_unet_2.sum()>0:
            x_2 = self.unet_2(x[timesteps_unet_2], timesteps[timesteps_unet_2], y[timesteps_unet_2])
            out[timesteps_unet_2] = x_2
        return out
