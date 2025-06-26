import collections
import numpy as np
import types
import torch
import torch.nn as nn
import torch.nn.functional as F


def snip_forward_conv2d(self, x):
    return F.conv2d(
        x,
        self.weight * self.weight_mask,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
    )


def snip_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)


class sparse_net_func(object):
    def __init__(self, keep_ratio, algo="IQL"):
        self.all_scores = None
        self.keep_ratio = keep_ratio
        self.algo = algo
        self.record_score = collections.defaultdict(dict)
        self.last_mask = collections.defaultdict(dict)

    def monkey_patch(self, net):
        for layer in net.modules():  # for name, layer in net.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
                layer.weight.requires_grad = False
            # Override the forward methods:
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(snip_forward_conv2d, layer)
            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(snip_forward_linear, layer)

    def get_keep_masks(self, net):
        grads_abs = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                grads_abs.append(torch.abs(layer.weight_mask.grad))
        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
        norm_factor = torch.sum(all_scores)
        all_scores.div_(norm_factor)
        num_params_to_keep = int(len(all_scores) * self.keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]
        if acceptable_score == 0:
            acceptable_score = self.last_acceptable_score
        else:
            self.last_acceptable_score = acceptable_score
        keep_masks = []
        for g in grads_abs:
            keep_masks.append(((g / norm_factor) >= acceptable_score).float())
        return keep_masks

    def get_masks(self, net, replay_buffer, batch_size=256, itr=1):
        if self.algo == "IQL":
            self.monkey_patch(net.actor)
            self.monkey_patch(net.critic)
            self.monkey_patch(net.value)
            self.monkey_patch(net.critic_target)
            net.train_no_grad_update(replay_buffer, itr, batch_size)
            return {
                "actor": (self.get_keep_masks(net.actor)),
                "critic": (self.get_keep_masks(net.critic)),
                "value": (self.get_keep_masks(net.value)),
            }

        elif self.algo == "AWAC":
            self.monkey_patch(net.actor)
            self.monkey_patch(net.critic)
            self.monkey_patch(net.critic_target)
            net.train_no_grad_update(replay_buffer, itr, batch_size)
            return {
                "actor": (self.get_keep_masks(net.actor)),
                "critic": (self.get_keep_masks(net.critic)),
            }

        elif self.algo == "TD3_BC":
            self.monkey_patch(net.actor)
            self.monkey_patch(net.critic)
            self.monkey_patch(net.actor_target)
            self.monkey_patch(net.critic_target)
            net.train_no_grad_update(replay_buffer, itr, batch_size)
            return {
                "actor": (self.get_keep_masks(net.actor)),
                "critic": (self.get_keep_masks(net.critic)),
            }

        elif self.algo == "BC":
            self.monkey_patch(net.actor)
            net.train_no_grad_update(replay_buffer, itr, batch_size)
            return {"actor": (self.get_keep_masks(net.actor))}
