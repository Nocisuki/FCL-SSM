import math
import os
from copy import deepcopy

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


class DeepInversionHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module, mmt_rate):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.mmt_rate = mmt_rate
        self.mmt = None
        self.tmp_val = None

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        if self.mmt is None:
            r_feature = torch.norm(module.running_var.data - var, 2) + \
                        torch.norm(module.running_mean.data - mean, 2)
        else:
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(module.running_var.data - (1 - self.mmt_rate) * var - self.mmt_rate * var_mmt, 2) + \
                        torch.norm(module.running_mean.data - (1 - self.mmt_rate) * mean - self.mmt_rate * mean_mmt, 2)

        self.r_feature = r_feature
        self.tmp_val = (mean, var)

    def update_mmt(self):
        mean, var = self.tmp_val
        if self.mmt is None:
            self.mmt = (mean.data, var.data)
        else:
            mean_mmt, var_mmt = self.mmt
            self.mmt = (self.mmt_rate * mean_mmt + (1 - self.mmt_rate) * mean.data,
                        self.mmt_rate * var_mmt + (1 - self.mmt_rate) * var.data)

    def remove(self):
        self.hook.remove()


# https://github.com/GT-RIPL/AlwaysBeDreaming-DFCIL/blob/main/learners/datafree_helper.py#L188
class Gaussiansmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma=1, dim=2):
        super().__init__()
        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                    1
                    / (std * math.sqrt(2 * math.pi))
                    * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1)).cuda()

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


def cumulative(lists):
    return [sum(lists[0:x:1]) for x in range(1, len(lists) + 1)]


def apply_prunning(cfg, task, trained_tasks):
    total_tasks = task | trained_tasks

    for classe in range(cfg["num_classes"]):
        if classe not in total_tasks:
            cfg["classifier"].fc.weight.data[classe] *= 0.0
            cfg["classifier"].fc.bias.data[classe] *= 0.0


def filter_logits(cfg, logits, task):
    mask = torch.ones(cfg["num_classes"], device=cfg["device"])

    for classe in range(cfg["num_classes"]):
        if classe not in task:
            mask[classe] *= 0.0

    with torch.no_grad():
        for i in range(len(logits)):
            logits[i] *= mask

    return logits


# def save_image_batch(save_syn_imgs, output, col=None, size=None):
#     if isinstance(save_syn_imgs, torch.Tensor):
#         save_syn_imgs = (save_syn_imgs.detach().clamp(0, 1).cpu().numpy() * 255).astype('uint8')
#     base_dir = os.path.dirname(output)
#     if base_dir != '':
#         os.makedirs(base_dir, exist_ok=True)
#
#     output_filename = output.strip('.png')
#     for idx, img in enumerate(save_syn_imgs):
#         img = Image.fromarray(img.transpose(1, 2, 0))
#         img.save(output_filename + '-%d.png' % (idx))

def save_image_batch(imgs, output, col=None, size=None):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy() * 255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir != '':
        os.makedirs(base_dir, exist_ok=True)

    output_filename = output.strip('.png')

    for idx, img in enumerate(imgs):

        if img.shape[0] == 1:
            img = img[0]
            img = Image.fromarray(img)
        else:
            img = Image.fromarray(img.transpose(1, 2, 0))

        # save
        img.save(f"{output_filename}-{idx}.png")