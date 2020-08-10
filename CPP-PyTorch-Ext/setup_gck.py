from setuptools import setup, Extension
from torch.utils import cpp_extension

# python setup_hw.py install

setup(name='gck_cpp',
      ext_modules=[
          cpp_extension.CppExtension(
              'gck_cpu_cpp', ['ConvImp.cpp', "CPP-PyTorch-Ext.cpp"])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})