#!/bin/bash
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file"
export IGC_VISAOptions="-perfmodel"
export IGC_VectorAliasBBThreshold=10000
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"

mkdir build_intel && cd build_intel

CC=icx CXX=icpx cmake .. -G Ninja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21 \
  -DCMAKE_CXX_FLAGS="-ftemplate-backtrace-limit=0 -fdiagnostics-color=always"
