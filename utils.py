from typing import Type

import torch
import torch.nn.functional as f
from torch import autograd
from mdmm import MDMM


def pad_to_match(source: torch.Tensor, reference: torch.Tensor):
    """Add right padding to a source tensor, to match the size of a reference
    tensor along dimension 1. Tensors are assumed to have shape (B, L, D)."""
    source_dim = source.shape[1]
    target_dim = reference.shape[1]
    pad = (0, 0, 0, target_dim - source_dim, 0, 0)
    return f.pad(source, pad)


def get_gradient_penalty(discriminator: torch.nn.Module,
                         real_samples: torch.Tensor,
                         fake_samples: torch.Tensor) -> torch.Tensor:
    """Get penalty for the gradient of the output of the discriminator with
    respect to its inputs. Inputs are linear interpolations between real
    and fake samples. The penalty is defined as the mean squared deviation
    of the norm from 1.

    Args:
        discriminator (torch.nn.Module): Module mapping a sample to a scalar
        real_samples (Tensor): shape (B, N, D)
        fake_samples (Tensor): shape (B, N, D)

    Returns:
        Tensor containing float with the mean norm of the gradient
    """
    device = real_samples.device

    # Make sizes equal along batch dimension by dropping samples
    if real_samples.shape[0] > fake_samples.shape[0]:
        real_samples = real_samples[:fake_samples.shape[0]]
    else:
        fake_samples = fake_samples[:real_samples.shape[0]]

    # Make sizes equal along sequence dimension by adding padding
    if real_samples.shape[1] < fake_samples.shape[1]:
        real_samples = pad_to_match(real_samples, fake_samples)
    else:
        fake_samples = pad_to_match(fake_samples, real_samples)

    a = torch.rand((fake_samples.shape[0], 1, 1), device=device)
    interpolates = a * real_samples + (1 - a) * fake_samples
    interpolates.requires_grad = True

    disc_interpolates = discriminator(interpolates)

    grad_outputs = torch.ones_like(disc_interpolates)
    disc_gradient = autograd.grad(disc_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=grad_outputs,
                                  create_graph=True,
                                  only_inputs=True)[0]

    grad_norm = disc_gradient.norm(p=2, dim=(1, 2))
    penalty = (grad_norm - 1).square().mean()

    return penalty


def make_mdmm_optimizer(mdmm_module: MDMM,
                        mdmm_lr: float,
                        optimizer_class: Type):
    lambdas = [c.lmbda for c in mdmm_module]
    slacks = [c.slack for c in mdmm_module if hasattr(c, 'slack')]
    return optimizer_class([{'params': lambdas, 'lr': -mdmm_lr},
                            {'params': slacks, 'lr': mdmm_lr}])
