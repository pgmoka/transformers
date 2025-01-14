#!/bin/bash
# cd transformers
pip3 install git+file://$PWD
pip3 install accelerate datasets evaluate scikit-learn huggingface-hub
pip3 install torch==2.7.0.dev20250113+cpu.cxx11.abi --index-url https://download.pytorch.org/whl/nightly
pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.7.0.dev20250113+cxx11-cp310-cp310-linux_x86_64.whl
pip install https://storage.googleapis.com/libtpu-nightly-releases/wheels/libtpu-nightly/libtpu_nightly-0.1.dev20250113+nightly-py3-none-linux_x86_64.whl
# install jax-0.4.36 jaxlib-0.4.36 ml_dtypes-0.5.0 opt_einsum-3.4.0
pip install jax==0.4.39.dev20250113 jaxlib==0.4.39.dev20250113 -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html