import sys
import numpy.linalg as l
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from gck_cpu_cpp import conv_fwd_3x3, matmul_only

''' 
    Data preperation (find linear combinations)
    -------------------------------------------
'''

def GCKBasisMatrix() -> (torch.Tensor):
    vecs = torch.Tensor([
        [1,1,1], [1,-1,1], [1,1,-1]
        ])
    basis = []
    for i in range(len(vecs)):
        for j in range(len(vecs)):
            mat = torch.ger(vecs[i], vecs[j])
            basis.append(mat)
    ret = torch.stack(basis)
    ret.requires_grad = False
    return ret
def LinCombPrepareBasis(basis, dtype=torch.float, verbose=False) -> torch.Tensor:
    # Basis shape is (9, 3, 3) (i.e 9 kernels, each is a 3x3 matrix)
    basis_flat = basis.reshape(9, 9).to(dtype)#.transpose(0,1)
    basis_flat_inv = torch.inverse(basis_flat)
    return basis_flat_inv
def LinCombWeights(basis_inv, weights, dtype=torch.float, verbose=False):
    w_mat = weights.reshape((len(weights), -1)).to(dtype).transpose(0,1)
    comb = basis_inv @ w_mat
    return torch.squeeze(comb, 1)
def ResultDim(input_dim, kernel_dim, pad=0, stride=1):
    s = math.floor((input_dim - kernel_dim + pad + pad) / stride) + 1
    return int(s)

class GCK3x3Layer(torch.nn.Module):
    def __init__(self, in_channels:int, out_channels:int, result_dim:int, kernel_size:int=3, bias:bool=False, padding:int=0, kernels:torch.Tensor=None):
        super(GCK3x3Layer, self).__init__()
        if (kernel_size != 3):
            raise "Supporting only 3x3 kernels"
        if (bias):
            raise "Not supporting bias"
        self.in_channels = in_channels
        self.result_dim = result_dim
        self.out_channels = out_channels
        self.padding = padding
        
        if kernels is None:
            kernels = torch.randn((out_channels, in_channels, 3, 3), dtype=torch.float32, requires_grad=False)

        basis = GCKBasisMatrix()
        basis_inv = LinCombPrepareBasis(basis)
        
        if (in_channels == 1):
            self.linCombs = LinCombWeights(LinCombPrepareBasis(basis), kernels)
            self.linCombs = self.linCombs.permute(1,0)
        else:
            k = torch.stack([LinCombWeights(basis_inv, kernels[i]) for i in range(len(kernels))])
            if (k.dim() != 3): k = k.unsqueeze(0)
            self.linCombs = k.permute(0,2,1).reshape(out_channels, in_channels * 9)
        
        self.linCombs = self.linCombs.contiguous()
        self.linCombs.requires_grad = False
        
        if (result_dim == 0):
            self.rowwiseResults = None
        else:
            self.rowwiseResults = torch.empty((in_channels * 3, result_dim*(result_dim+2))).contiguous()
            self.colwiseResults = torch.empty((in_channels * 9, result_dim*result_dim)).contiguous()

    def forward(self, input: torch.Tensor):
        if (self.rowwiseResults == None):
            if (self.padding == 0):
                result_dim = input.shape[-1]-2
            else:
                result_dim = input.shape[-1]
            self.result_dim = result_dim
            self.rowwiseResults = torch.empty((self.in_channels * 3, result_dim*(result_dim+2))).contiguous()
            self.colwiseResults = torch.empty((self.in_channels * 9, result_dim*result_dim)).contiguous()
        
        result = conv_fwd_3x3(input, self.linCombs, self.rowwiseResults, self.colwiseResults, self.padding)
        
        return result
    
    def matmul_only(self, input: torch.Tensor):
        return matmul_only(input, self.linCombs, self.rowwiseResults, self.colwiseResults, self.padding)
    
    def extra_repr(self):
        return f'FastConv({self.in_channels}, {self.out_channels}, kernel_size=(3,3), stride=(1,1), padding=({self.padding},{self.padding}), bias=False, result_dim={self.result_dim}';