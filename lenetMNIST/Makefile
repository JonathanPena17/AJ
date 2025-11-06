CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
NVCC_VER_REQ = 11.0
NVCC_VER = $(shell $(NVCC) --version | grep release | sed 's/.*release //' | sed 's/,.*//')

ARCH ?= -gencode arch=compute_60,code=sm_60 \
        -gencode arch=compute_70,code=sm_70 \
        -gencode arch=compute_75,code=sm_75 \
        -gencode arch=compute_80,code=sm_80 \
        -gencode arch=compute_86,code=sm_86

NVCCFLAGS = -O3 -std=c++11 $(ARCH)

TARGET = lenet5_mnist
SOURCE = lenet5_mnist_train.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) $(SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET)
golden:
	./lenet5_mnist train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte >golden_stdout.txt 2>golden_stderr.txt

.PHONY: all clean

