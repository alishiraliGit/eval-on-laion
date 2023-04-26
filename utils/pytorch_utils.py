from contextlib import contextmanager

import torch
from torch import nn


device = None


def init_gpu(use_gpu=True, gpu_id=0, verbose=True):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device('cuda:' + str(gpu_id))
        if verbose:
            print('\tusing gpu id {}'.format(gpu_id))
    else:
        device = torch.device('cpu')
        if verbose:
            print('\tgpu not detected, defaulting to cpu.')


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


@contextmanager
def eval_mode(net: nn.Module):
    """Temporarily switch to evaluation mode."""
    originally_training = net.training
    try:
        net.eval()
        yield net
    finally:
        if originally_training:
            net.train()
