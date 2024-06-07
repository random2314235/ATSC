from __future__ import print_function

import json
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist


def extract_weights_except_last(model, opt, detach = True):
    # This dictionary will hold the weights of all layers except the last one
    weights = {}

    # Get all parameters and names
    all_params = list(model.named_parameters())

    # Exclude the last parameter, assuming it belongs to the last layer
    for name, param in all_params[:-1]:
        # Check if the parameter is a weight (often weights have more than 1 dimension)
        if param.requires_grad and param.ndim > 1:
            if detach:
                weights[name] = param.clone().detach().cuda(opt.gpu)
            else:
                weights[name] = param
    return weights

def sum_of_weight_differences(weight_dict1, weight_dict2):
    total_difference = 0

    for key in weight_dict1:
        # Calculate the difference for the corresponding parameter
        difference = weight_dict1[key] - weight_dict2[key]
        # Sum the absolute values of the differences
        total_difference += torch.sum(torch.square(difference))

    return total_difference

def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

# def adjust_learning_rate(optimizer, epoch, step, len_epoch, old_lr):
#     """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
#     if epoch < 5:
#         lr = old_lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)
#     elif 5 <= epoch < 60: return
#     else:
#         factor = epoch // 30
#         factor -= 1
#         lr = old_lr*(0.1**factor)

#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim = 1, largest = True, sorted = True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)

def load_json_to_dict(json_path):
    """Loads json file to dict 

    Args:
        json_path: (string) path to json file
    """
    with open(json_path, 'r') as f:
        params = json.load(f)
    return params

def reduce_tensor(tensor, world_size = 1, op='avg'):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size > 1:
        rt = torch.true_divide(rt, world_size)
    return rt

if __name__ == '__main__':

    pass
