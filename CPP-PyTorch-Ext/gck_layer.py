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
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, bias:bool, result_dim:int, kernels:torch.Tensor=None):
        self.out_channels = out_channels
        
        if kernels is None:
            kernels = torch.randn((out_channels, in_channels, kernel_size, kernel_size), dtype=torch.float32, requires_grad=False)

        basis = GCKBasisMatrix()
        basis_inv = LinCombPrepareBasis(basis)
        
        if (in_channels == 1):
            self.linCombs = LinCombWeights(LinCombPrepareBasis(basis), kernels)
            self.linCombs = self.linCombs.permute(1,0)
        else:
            k = torch.stack([LinCombWeights(basis_inv, kernels[i]) for i in range(len(kernels))])
            #print('b', k.shape)
            if (k.dim() != 3): k = k.unsqueeze(0)
            self.linCombs = k.permute(0,2,1).reshape(out_channels, in_channels * 9)
            #print('c', k.shape)
        
        self.linCombs = self.linCombs.contiguous()
        self.linCombs.requires_grad = False
        
        self.rowwiseResults = torch.empty((in_channels * 3, result_dim*(result_dim+2))).contiguous()
        self.colwiseResults = torch.empty((in_channels * 9, result_dim*result_dim)).contiguous()
    def forward(self, input: torch.Tensor):
        result = conv_fwd_3x3(input, self.linCombs, self.rowwiseResults, self.colwiseResults)
        return result
    def matmul_only(self, input: torch.Tensor):
        return matmul_only(input, self.linCombs, self.rowwiseResults, self.colwiseResults)