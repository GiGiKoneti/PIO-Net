#!/bin/bash
# Build CMSIS-NN BNN example for Cortex-M0+ and run under QEMU.
# This is a placeholder; actual steps require ARM toolchain and CMSIS-NN repo.

set -e

CMSIS_NN_DIR=${CMSIS_NN_DIR:-../CMSIS-NN}
BUILD_DIR=build_cmsisnn
EXAMPLE=Examples/BinaryNeuralNetwork

mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/arm-none-eabi.cmake \
      -DCMSIS_NN_ROOT=$CMSIS_NN_DIR \
      $CMSIS_NN_DIR/$EXAMPLE

make -j$(nproc)

# Run under QEMU (machine virt, cortex-m0plus)
qemu-system-arm -M virt -cpu cortex-m0plus -nographic \
    -kernel ${EXAMPLE}.elf -semihosting-config enable=on,target=native \
    -chardev stdio,id=stdio,mux=on -serial chardev:stdio \
    -monitor none

# The example should print cycle counts via semihosting; capture and parse.