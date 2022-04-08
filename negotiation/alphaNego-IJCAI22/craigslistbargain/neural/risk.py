import numpy as np
import torch

import onmt.pytorch_utils as ptu


def normal_cdf(value, loc=0., scale=1.):
    return 0.5 * (1 + torch.erf((value - loc) / scale / np.sqrt(2)))


def normal_icdf(value, loc=0., scale=1.):
    return loc + scale * torch.erfinv(2 * value - 1) * np.sqrt(2)


def normal_pdf(value, loc=0., scale=1.):
    return torch.exp(-(value - loc)**2 / (2 * scale**2)) / scale / np.sqrt(2 * np.pi)


def distortion_fn(tau, mode="neutral", param=0.):
    # Risk distortion function
    tau = tau.clamp(0., 1.)
    if param >= 0:
        if mode == "neutral":
            tau_ = tau
        elif mode == "wang":
            tau_ = normal_cdf(normal_icdf(tau) + param)
        elif mode == "cvar":
            tau_ = (1. / param) * tau
        elif mode == "cpw":
            tau_ = tau**param / (tau**param + (1. - tau)**param)**(1. / param)
        return tau_.clamp(0., 1.)
    else:
        return 1 - distortion_fn(1 - tau, mode, -param)


def distortion_de(tau, mode="neutral", param=0., eps=1e-8):
    # Derivative of Risk distortion function
    tau = tau.clamp(0., 1.)
    if param >= 0:
        if mode == "neutral":
            tau_ = ptu.one_like(tau)
        elif mode == "wang":
            tau_ = normal_pdf(normal_icdf(tau) + param) / (normal_pdf(normal_icdf(tau)) + eps)
        elif mode == "cvar":
            tau_ = (1. / param) * (tau < param)
        elif mode == "cpw":
            g = tau**param
            h = (tau**param + (1 - tau)**param)**(1 / param)
            g_ = param * tau**(param - 1)
            h_ = (tau**param + (1 - tau)**param)**(1 / param - 1) * (tau**(param - 1) - (1 - tau)**(param - 1))
            tau_ = (g_ * h - g * h_) / (h**2 + eps)
        return tau_.clamp(0., 5.)

    else:
        return distortion_de(1 - tau, mode, -param)
