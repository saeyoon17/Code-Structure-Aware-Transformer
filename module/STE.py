import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SampleGraphSparseGraph"]


class SampleGraphSparseGraph(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        tmp = input.clamp(0.01, 0.99)
        tmp.requires_grad_(True)
        A = torch.bernoulli(tmp)
        ctx.save_for_backward(A)
        return A

    def backward(ctx, grad_output):
        (A,) = ctx.saved_tensors
        return F.hardtanh(A * grad_output)
