import sys
import numpy.linalg as l
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gck_cpu_cpp

def GCKBasisMatrix() -> (torch.Tensor):
    vec = torch.Tensor([
        [1,1,1], [1,-1,1], [1,1,-1]
        ])
    basis = []
    for i in range(len(vecs)):
        for j in range(len(vecs)):
            mat = torch.ger(vecs[i], vecs[j])
            basis.append(mat)
      #ret = torch.Tensor([
      #    [[1,1,1], [1,1,1], [1,1,1]],
      #    [[1,-1,1], [
      #    [[1,1,1], [-1,-1,-1], [1,1,1]],
      #    [[1,1,1], [1,1,1], [-1,-1,-1]],

      #        ])
    ret = torch.stack(basis)
    ret.requires_grad = False
    return ret
''' Finds comb '''
def linCombPrepareBasis(basis, dtype=torch.float, verbose=False) -> torch.Tensor:
  B = len(basis) ** 2 # Basis shape is (Rows, Cols, basis_rows, basis_cols). Rows == Cols, basis_rows == basis_cols so Rows^2==Rows*Cols
  basis_flat = basis.reshape(B, -1).transpose(0,1).to(dtype)
  basis_flat_inv = torch.inverse(basis_flat)
  return basis_flat_inv
def linCombWeights(basis_inv, weights, dtype=torch.float, verbose=False):
  w_mat = weights.reshape((len(weights), -1)).transpose(0,1).to(dtype)
  comb = basis_inv @ w_mat
  return torch.squeeze(comb, 1)
def resultDim(input_dim, kernel_dim, pad=0, stride=1):
    s = math.floor((input_dim - kernel_dim + pad + pad) / stride) + 1
    return int(s)

class GCK3x3Layer(torch.nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, bias:bool, result_dim:int, kernels:torch.Tensor=None):
        #self.basis, self.prefixes, self.positive_signs = GCKBasisMatrix()
        self.out_channels = out_channels
        
        #if kernels is None:
        #    kernels = torch.randn((out_channels, in_channels, kernel_size, kernel_size), dtype=torch.float32, requires_grad=False)
        #basis_inv = linCombPrepareBasis(self.basis)
        #k = torch.stack([linCombWeights(basis_inv, kernels[i]) for i in range(len(kernels))])
        #if (k.dim() != 3): k = k.unsqueeze(0)
        #k = k.permute(0,2,1).reshape(out_channels, in_channels * 9)
        self.kernelsL = torch.randn(out_channels, in_channels * 9)
        self.kernelsL.requires_grad = False
        self.kernels = kernels

        self.helper = torch.empty((in_channels * 9, result_dim*result_dim)).contiguous()

    def forward(self, input: torch.Tensor):
        #out = torch.empty(input.size(0), self.kernelsL.size(0), input.size(2), input.size(3))
        #for batch_ix in range(input.size(0)):
        #    a = torch.ones(self.kernelsL.size(1), input.size(2) * input.size(3))
        #    c = torch.mm(self.kernelsL, a,out=out[batch_ix])
        #result = out 
        #print('pre', self.kernelsL.shape, self.helper.shape)
        result = gck_cpu_cpp.conv_fwd_3x3(input, self.kernelsL, self.helper)
        return result

    def convolveWithBasisRegular(self, input):
        print('input shape', input.shape)
        batch, in_channels, height, width = input.shape
        if (batch != 1): raise "Only batch=1 is supported"
        stride=1
        padding=0
        basis_reg_conv = self.basis.reshape(-1, 1, 4, 4)
        inp_reshaped = input.reshape(-1, 1, height, width) # Convert in channels to batch (looked separatly)
        after_conv = F.conv2d(inp_reshaped, basis_reg_conv, stride=stride, padding=padding)
        after_conv = after_conv.reshape(in_channels * 16, after_conv.size(2), after_conv.size(3)) # Join the batches
        after_conv = after_conv.flatten(start_dim=1)
        result = torch.mm(self.kernelsL, after_conv)
        result = result.reshape(1, self.kernelsL.size(0), resultDim(height, 4), resultDim(width, 4))
        print('output shape', result.shape)
        return result
    def convolveWithBasisGCK(self, input):
        print('input shape', input.shape)
        batch, in_channels, height, width = input.shape
        if (batch != 1): raise "Only batch=1 is supported"

        after_conv = gck_cpu_cpp.conv_basis_4x4(input, self.prefixes, self.positive_signs)
        after_conv = after_conv.squeeze(0).flatten(start_dim=1)
        print(after_conv.shape)
        result = torch.mm(self.kernelsL, after_conv)
        result = result.reshape(1, self.kernelsL.size(0), resultDim(height, 4), resultDim(width, 4))
        print('output shape', result.shape)
        return result
    def compareBasisConv(self, input):
        print('input shape', input.shape)
        batch, in_channels, height, width = input.shape
        if (batch != 1): raise "Only batch=1 is supported"
        stride=1
        padding=0
        basis_reg_conv = self.basis.reshape(-1, 1, 4, 4)
        inp_reshaped = input.reshape(-1, 1, height, width) # Convert in channels to batch (looked separatly)
        after_conv = F.conv2d(inp_reshaped, basis_reg_conv, stride=stride, padding=padding)
        after_conv = after_conv.reshape(in_channels * 16, after_conv.size(2), after_conv.size(3)) # Join the batches
        
        after_conv_gck = gck_cpu_cpp.conv_basis_4x4(input, self.prefixes, self.positive_signs).squeeze(0)

        for i in range(64):
            #print(after_conv[i])
            #print(after_conv_gck[i])
            print(torch.allclose(after_conv[i], after_conv_gck[i], atol=0.01))
    def compareWithPyTorch(self,input):
        after_conv_reshaped = after_conv.reshape(batch, in_channels, self.kernel_squared, after_conv.shape[2], after_conv.shape[3])
        gck_cpp.gck_forward_basis(input, self.basis_flattened, self.basis_vectors, convResult, self.kernel_size, convResult.shape[3], convResult.shape[3]**2, self.prefixes_flattened, self.positive_signs_flattened)

        for i in range(16):
            diff = (x[0][0][i] - after_conv_reshaped[0][0][i]).max()
            print(basis_reg_conv[i][0])
            if (abs(diff) > 1):
                print('error at kernel ', i, diff)
                print(x[0][0][i])
                print(after_conv_reshaped[0][0][i])
            else:
                print('kernel', i, 'equals')