#!/bin/bash

# Change to the directory where this script is located
#cd "$(dirname "$0")" || exit

# Run LeNet-5 MNIST training

# Set defaults if not provided by NVBitFI
BIN_DIR=${BIN_DIR:-.}
PRELOAD_FLAG=${PRELOAD_FLAG:-""}

eval ${PRELOAD_FLAG} ${BIN_DIR}/lenet5_mnist \
    /home/ubuntu/nvbit_release/tools/nvbitfi/test-apps/lenetMNIST/train-images-idx3-ubyte \
    /home/ubuntu/nvbit_release/tools/nvbitfi/test-apps/lenetMNIST/train-labels-idx1-ubyte \
    /home/ubuntu/nvbit_release/tools/nvbitfi/test-apps/lenetMNIST/t10k-images-idx3-ubyte \
    /home/ubuntu/nvbit_release/tools/nvbitfi/test-apps/lenetMNIST/t10k-labels-idx1-ubyte \
    1>stdout.txt 2>stderr.txt

