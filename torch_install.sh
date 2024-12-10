#!/bin/bash
cd transformers
pip3 install git+file://$PWD
pip3 install accelerate datasets evaluate scikit-learn huggingface-hub
pip3 install torch==2.6.0.dev20241209+cpu.cxx11.abi --index-url https://download.pytorch.org/whl/nightly
wget https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0.dev20241209.cxx11-cp310-cp310-linux_x86_64.whl
mv torch_xla-2.6.0.dev20241209.cxx11-cp310-cp310-linux_x86_64.whl torch_xla-2.6.0.dev20241209-cp310-cp310-linux_x86_64.whl
pip3 install torch_xla-2.6.0.dev20241209-cp310-cp310-linux_x86_64.whl
pip3 install https://storage.googleapis.com/libtpu-lts-releases/wheels/libtpu/libtpu-0.0.5-py3-none-linux_x86_64.whl

# install jax-0.4.36 jaxlib-0.4.36 ml_dtypes-0.5.0 opt_einsum-3.4.0
pip install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html