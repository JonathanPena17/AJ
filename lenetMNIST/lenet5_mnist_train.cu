#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdint.h> 

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define INPUT_SIZE 28
#define INPUT_CHANNELS 1  
#define C1_FILTERS 6
#define C1_SIZE 24       
#define S2_SIZE 12       
#define C3_FILTERS 16
#define C3_SIZE 8        
#define S4_SIZE 4        
#define C5_SIZE 120
#define F6_SIZE 84
#define OUTPUT_SIZE 10
#define CONV_KERNEL 5
#define POOL_SIZE 2

// Training hyperparameters
#define BATCH_SIZE 64
#define LEARNING_RATE 0.00001f
#define EPOCHS 20
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

// ==================== MNIST Data Loading ====================

typedef struct {
    float* images;     
    unsigned char* labels;
    int count;
} MNISTData;


uint32_t reverse_int(uint32_t i) {
    uint8_t c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
}

// Load MNIST images
bool load_mnist_images(const char* filename, float** images, int* num_images) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return false;
    }

    uint32_t magic_number = 0;
    uint32_t num_imgs = 0;
    uint32_t num_rows = 0;
    uint32_t num_cols = 0;

    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = reverse_int(magic_number);
    
    if (magic_number != 2051) {
        fprintf(stderr, "Invalid MNIST image file!\n");
        fclose(file);
        return false;
    }

    fread(&num_imgs, sizeof(num_imgs), 1, file);
    num_imgs = reverse_int(num_imgs);
    fread(&num_rows, sizeof(num_rows), 1, file);
    num_rows = reverse_int(num_rows);
    fread(&num_cols, sizeof(num_cols), 1, file);
    num_cols = reverse_int(num_cols);

    *num_images = num_imgs;
    *images = (float*)malloc(num_imgs * 28 * 28 * sizeof(float));

    for (int i = 0; i < num_imgs; i++) {
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                unsigned char pixel = 0;
                fread(&pixel, sizeof(pixel), 1, file);
                (*images)[i * 28 * 28 + r * 28 + c] = pixel / 255.0f;
            }
        }
    }

    fclose(file);
    printf("Loaded %d images from %s\n", num_imgs, filename);
    return true;
}

// Load MNIST labels
bool load_mnist_labels(const char* filename, unsigned char** labels, int* num_labels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return false;
    }

    uint32_t magic_number = 0;
    uint32_t num_lbls = 0;

    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = reverse_int(magic_number);
    
    if (magic_number != 2049) {
        fprintf(stderr, "Invalid MNIST label file!\n");
        fclose(file);
        return false;
    }

    fread(&num_lbls, sizeof(num_lbls), 1, file);
    num_lbls = reverse_int(num_lbls);
    *num_labels = num_lbls;
    *labels = (unsigned char*)malloc(num_lbls);

    fread(*labels, sizeof(unsigned char), num_lbls, file);

    fclose(file);
    printf("Loaded %d labels from %s\n", num_lbls, filename);
    return true;
}

// Load MNIST training data
MNISTData load_mnist_train(const char* image_file, const char* label_file) {
    MNISTData data;
    if (!load_mnist_images(image_file, &data.images, &data.count)) {
        exit(1);
    }
    int label_count;
    if (!load_mnist_labels(label_file, &data.labels, &label_count)) {
        exit(1);
    }
    if (data.count != label_count) {
        fprintf(stderr, "Image and label count mismatch!\n");
        exit(1);
    }
    return data;
}

// Load MNIST test data
MNISTData load_mnist_test(const char* image_file, const char* label_file) {
    return load_mnist_train(image_file, label_file);
}

// ==================== CUDA Kernels ====================

// ReLU activation
__global__ void relu_forward(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// Convolution forward pass
__global__ void conv2d_forward(
    const float* input, const float* weight, const float* bias,
    float* output, int in_channels, int out_channels,
    int input_size, int output_size, int kernel_size
) {
    int out_ch = blockIdx.x;
    int out_y = blockIdx.y;
    int out_x = threadIdx.x;
    
    if (out_ch >= out_channels || out_y >= output_size || out_x >= output_size) return;
    
    float sum = 0.0f;
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int in_y = out_y + ky;
                int in_x = out_x + kx;
                int in_idx = in_ch * input_size * input_size + in_y * input_size + in_x;
                int w_idx = ((out_ch * in_channels + in_ch) * kernel_size + ky) * kernel_size + kx;
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }
    
    if (bias != NULL) sum += bias[out_ch];
    
    int out_idx = out_ch * output_size * output_size + out_y * output_size + out_x;
    output[out_idx] = sum;
}

// Max pooling forward
__global__ void maxpool2d_forward(
    const float* input, float* output, int* indices,
    int channels, int input_size, int output_size
) {
    int ch = blockIdx.x;
    int out_y = blockIdx.y;
    int out_x = threadIdx.x;
    
    if (ch >= channels || out_y >= output_size || out_x >= output_size) return;
    
    int in_y = out_y * 2;
    int in_x = out_x * 2;
    
    float max_val = -1e10f;
    int max_idx = 0;
    
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int in_idx = ch * input_size * input_size + (in_y + dy) * input_size + (in_x + dx);
            if (input[in_idx] > max_val) {
                max_val = input[in_idx];
                max_idx = dy * 2 + dx;
            }
        }
    }
    
    int out_idx = ch * output_size * output_size + out_y * output_size + out_x;
    output[out_idx] = max_val;
    if (indices) indices[out_idx] = max_idx;
}

// Fully connected forward
__global__ void fc_forward(
    const float* input, const float* weight, const float* bias,
    float* output, int input_size, int output_size
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= output_size) return;
    
    float sum = 0.0f;
    for (int i = 0; i < input_size; i++) {
        sum += input[i] * weight[out_idx * input_size + i];
    }
    if (bias != NULL) sum += bias[out_idx];
    output[out_idx] = sum;
}

// Softmax
__global__ void softmax_forward(float* data, int size) {
    __shared__ float max_val;
    __shared__ float sum_exp;
    
    if (threadIdx.x == 0) {
        max_val = data[0];
        for (int i = 1; i < size; i++) {
            max_val = fmaxf(max_val, data[i]);
        }
    }
    __syncthreads();
    
    int idx = threadIdx.x;
    if (idx < size) {
        data[idx] = expf(data[idx] - max_val);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        sum_exp = 0.0f;
        for (int i = 0; i < size; i++) {
            sum_exp += data[i];
        }
    }
    __syncthreads();
    
    if (idx < size) {
        data[idx] /= sum_exp;
    }
}

// Gradient kernels
__global__ void fc_backward_weight(
    const float* input, const float* grad_output,
    float* grad_weight, int input_size, int output_size, float lr
) {
    int out_idx = blockIdx.x;
    int in_idx = threadIdx.x;
    
    if (out_idx >= output_size || in_idx >= input_size) return;
    
    int w_idx = out_idx * input_size + in_idx;
    atomicAdd(&grad_weight[w_idx], -lr * grad_output[out_idx] * input[in_idx]);
}

__global__ void fc_backward_bias(
    const float* grad_output, float* bias, int output_size, float lr
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;
    atomicAdd(&bias[idx], -lr * grad_output[idx]);
}

__global__ void fc_backward_input(
    const float* grad_output, const float* weight,
    float* grad_input, int input_size, int output_size
) {
    int in_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (in_idx >= input_size) return;
    
    float sum = 0.0f;
    for (int out_idx = 0; out_idx < output_size; out_idx++) {
        sum += grad_output[out_idx] * weight[out_idx * input_size + in_idx];
    }
    grad_input[in_idx] = sum;
}

// Convolution backward - compute weight gradients
__global__ void conv2d_backward_weight(
    const float* input, const float* grad_output,
    float* grad_weight, int in_channels, int out_channels,
    int input_size, int output_size, int kernel_size, float lr
) {
    int out_ch = blockIdx.x;
    int in_ch = blockIdx.y;
    int ky = blockIdx.z;
    int kx = threadIdx.x;
    
    if (out_ch >= out_channels || in_ch >= in_channels || 
        ky >= kernel_size || kx >= kernel_size) return;
    
    float grad_sum = 0.0f;
    
    for (int out_y = 0; out_y < output_size; out_y++) {
        for (int out_x = 0; out_x < output_size; out_x++) {
            int in_y = out_y + ky;
            int in_x = out_x + kx;
            
            int in_idx = in_ch * input_size * input_size + in_y * input_size + in_x;
            int out_idx = out_ch * output_size * output_size + out_y * output_size + out_x;
            
            grad_sum += input[in_idx] * grad_output[out_idx];
        }
    }
    
    int w_idx = ((out_ch * in_channels + in_ch) * kernel_size + ky) * kernel_size + kx;
    atomicAdd(&grad_weight[w_idx], -lr * grad_sum);
}

// Convolution backward - compute bias gradients
__global__ void conv2d_backward_bias(
    const float* grad_output, float* bias,
    int out_channels, int output_size, float lr
) {
    int out_ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_ch >= out_channels) return;
    
    float grad_sum = 0.0f;
    for (int y = 0; y < output_size; y++) {
        for (int x = 0; x < output_size; x++) {
            int idx = out_ch * output_size * output_size + y * output_size + x;
            grad_sum += grad_output[idx];
        }
    }
    
    atomicAdd(&bias[out_ch], -lr * grad_sum);
}

// Convolution backward - propagate gradients to input
__global__ void conv2d_backward_input(
    const float* grad_output, const float* weight, float* grad_input,
    int in_channels, int out_channels, int input_size, int output_size, int kernel_size
) {
    int in_ch = blockIdx.x;
    int in_y = blockIdx.y;
    int in_x = threadIdx.x;
    
    if (in_ch >= in_channels || in_y >= input_size || in_x >= input_size) return;
    
    float grad_sum = 0.0f;
    
    for (int out_ch = 0; out_ch < out_channels; out_ch++) {
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int out_y = in_y - ky;
                int out_x = in_x - kx;
                
                if (out_y >= 0 && out_y < output_size && out_x >= 0 && out_x < output_size) {
                    int out_idx = out_ch * output_size * output_size + out_y * output_size + out_x;
                    int w_idx = ((out_ch * in_channels + in_ch) * kernel_size + ky) * kernel_size + kx;
                    grad_sum += grad_output[out_idx] * weight[w_idx];
                }
            }
        }
    }
    
    int in_idx = in_ch * input_size * input_size + in_y * input_size + in_x;
    grad_input[in_idx] = grad_sum;
}

// Max pooling backward
__global__ void maxpool2d_backward(
    const float* grad_output, const int* indices, float* grad_input,
    int channels, int input_size, int output_size
) {
    int ch = blockIdx.x;
    int out_y = blockIdx.y;
    int out_x = threadIdx.x;
    
    if (ch >= channels || out_y >= output_size || out_x >= output_size) return;
    
    int out_idx = ch * output_size * output_size + out_y * output_size + out_x;
    int max_idx = indices[out_idx];
    
    int dy = max_idx / 2;
    int dx = max_idx % 2;
    int in_y = out_y * 2 + dy;
    int in_x = out_x * 2 + dx;
    
    int in_idx = ch * input_size * input_size + in_y * input_size + in_x;
    atomicAdd(&grad_input[in_idx], grad_output[out_idx]);
}

// ReLU backward
__global__ void relu_backward(float* grad, const float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] *= (output[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

// ==================== LeNet-5 Model ====================
typedef struct {
    // Layer outputs (forward)
    float *d_input;  
    float *d_conv1_out, *d_pool1_out;
    float *d_conv2_out, *d_pool2_out;
    float *d_fc1_out, *d_fc2_out, *d_output;
    
    // Weights
    float *d_conv1_w, *d_conv1_b; 
    float *d_conv2_w, *d_conv2_b; 
    float *d_fc1_w, *d_fc1_b;     
    float *d_fc2_w, *d_fc2_b;     
    float *d_fc3_w, *d_fc3_b;      
    
    // Gradients (for backprop)
    float *d_grad_output, *d_grad_fc2, *d_grad_fc1;
    float *d_grad_pool2, *d_grad_conv2;
    float *d_grad_pool1, *d_grad_conv1;
    
    // Max pooling indices
    int *d_pool1_indices, *d_pool2_indices;
} LeNet5;

void init_lenet5(LeNet5* model) {
    // Allocate all memory
    CHECK_CUDA(cudaMalloc(&model->d_input, INPUT_CHANNELS * INPUT_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_conv1_out, C1_FILTERS * C1_SIZE * C1_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_pool1_out, C1_FILTERS * S2_SIZE * S2_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_conv2_out, C3_FILTERS * C3_SIZE * C3_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_pool2_out, C3_FILTERS * S4_SIZE * S4_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_fc1_out, C5_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_fc2_out, F6_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_output, OUTPUT_SIZE * sizeof(float)));
    
    // Weights - Conv1 now has 1 input channel for grayscale
    CHECK_CUDA(cudaMalloc(&model->d_conv1_w, C1_FILTERS * INPUT_CHANNELS * CONV_KERNEL * CONV_KERNEL * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_conv1_b, C1_FILTERS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_conv2_w, C3_FILTERS * C1_FILTERS * CONV_KERNEL * CONV_KERNEL * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_conv2_b, C3_FILTERS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_fc1_w, C5_SIZE * (C3_FILTERS * S4_SIZE * S4_SIZE) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_fc1_b, C5_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_fc2_w, F6_SIZE * C5_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_fc2_b, F6_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_fc3_w, OUTPUT_SIZE * F6_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_fc3_b, OUTPUT_SIZE * sizeof(float)));
    
    // Gradients
    CHECK_CUDA(cudaMalloc(&model->d_grad_output, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_grad_fc2, F6_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_grad_fc1, C5_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_grad_pool2, C3_FILTERS * S4_SIZE * S4_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_grad_conv2, C3_FILTERS * C3_SIZE * C3_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_grad_pool1, C1_FILTERS * S2_SIZE * S2_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_grad_conv1, C1_FILTERS * C1_SIZE * C1_SIZE * sizeof(float)));
    
    // Pooling indices
    CHECK_CUDA(cudaMalloc(&model->d_pool1_indices, C1_FILTERS * S2_SIZE * S2_SIZE * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&model->d_pool2_indices, C3_FILTERS * S4_SIZE * S4_SIZE * sizeof(int)));
    
    // Initialize weights with Xavier initialization
    srand(time(NULL));
    auto init_weights = [](float* d_ptr, int size, float scale) {
        float* h_w = (float*)malloc(size * sizeof(float));
        for (int i = 0; i < size; i++) {
            h_w[i] = (((float)rand() / RAND_MAX) - 0.5f) * 2.0f * scale;
        }
        CHECK_CUDA(cudaMemcpy(d_ptr, h_w, size * sizeof(float), cudaMemcpyHostToDevice));
        free(h_w);
    };
    
    // Xavier init for grayscale input
    init_weights(model->d_conv1_w, C1_FILTERS * INPUT_CHANNELS * CONV_KERNEL * CONV_KERNEL, 
                 sqrtf(2.0f / (INPUT_CHANNELS * CONV_KERNEL * CONV_KERNEL)));
    init_weights(model->d_conv2_w, C3_FILTERS * C1_FILTERS * CONV_KERNEL * CONV_KERNEL, 
                 sqrtf(2.0f / (C1_FILTERS * CONV_KERNEL * CONV_KERNEL)));
    init_weights(model->d_fc1_w, C5_SIZE * C3_FILTERS * S4_SIZE * S4_SIZE, 
                 sqrtf(2.0f / (C3_FILTERS * S4_SIZE * S4_SIZE)));
    init_weights(model->d_fc2_w, F6_SIZE * C5_SIZE, sqrtf(2.0f / C5_SIZE));
    init_weights(model->d_fc3_w, OUTPUT_SIZE * F6_SIZE, sqrtf(2.0f / F6_SIZE));
    
    CHECK_CUDA(cudaMemset(model->d_conv1_b, 0, C1_FILTERS * sizeof(float)));
    CHECK_CUDA(cudaMemset(model->d_conv2_b, 0, C3_FILTERS * sizeof(float)));
    CHECK_CUDA(cudaMemset(model->d_fc1_b, 0, C5_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemset(model->d_fc2_b, 0, F6_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemset(model->d_fc3_b, 0, OUTPUT_SIZE * sizeof(float)));
}

void forward_pass(LeNet5* model) {
    // Conv1: 28x28x1 -> 24x24x6
    dim3 grid1(C1_FILTERS, C1_SIZE, 1);
    conv2d_forward<<<grid1, C1_SIZE>>>(
        model->d_input, model->d_conv1_w, model->d_conv1_b,
        model->d_conv1_out, INPUT_CHANNELS, C1_FILTERS, INPUT_SIZE, C1_SIZE, CONV_KERNEL
    );
    relu_forward<<<(C1_FILTERS * C1_SIZE * C1_SIZE + 255) / 256, 256>>>(
        model->d_conv1_out, C1_FILTERS * C1_SIZE * C1_SIZE
    );
    
    // Pool1: 24x24x6 -> 12x12x6
    dim3 grid2(C1_FILTERS, S2_SIZE, 1);
    maxpool2d_forward<<<grid2, S2_SIZE>>>(
        model->d_conv1_out, model->d_pool1_out, model->d_pool1_indices,
        C1_FILTERS, C1_SIZE, S2_SIZE
    );
    
    // Conv2: 12x12x6 -> 8x8x16
    dim3 grid3(C3_FILTERS, C3_SIZE, 1);
    conv2d_forward<<<grid3, C3_SIZE>>>(
        model->d_pool1_out, model->d_conv2_w, model->d_conv2_b,
        model->d_conv2_out, C1_FILTERS, C3_FILTERS, S2_SIZE, C3_SIZE, CONV_KERNEL
    );
    relu_forward<<<(C3_FILTERS * C3_SIZE * C3_SIZE + 255) / 256, 256>>>(
        model->d_conv2_out, C3_FILTERS * C3_SIZE * C3_SIZE
    );
    
    // Pool2: 8x8x16 -> 4x4x16
    dim3 grid4(C3_FILTERS, S4_SIZE, 1);
    maxpool2d_forward<<<grid4, S4_SIZE>>>(
        model->d_conv2_out, model->d_pool2_out, model->d_pool2_indices,
        C3_FILTERS, C3_SIZE, S4_SIZE
    );
    
    // FC1: 256 -> 120
    fc_forward<<<(C5_SIZE + 255) / 256, 256>>>(
        model->d_pool2_out, model->d_fc1_w, model->d_fc1_b,
        model->d_fc1_out, C3_FILTERS * S4_SIZE * S4_SIZE, C5_SIZE
    );
    relu_forward<<<(C5_SIZE + 255) / 256, 256>>>(model->d_fc1_out, C5_SIZE);
    
    // FC2: 120 -> 84
    fc_forward<<<(F6_SIZE + 255) / 256, 256>>>(
        model->d_fc1_out, model->d_fc2_w, model->d_fc2_b,
        model->d_fc2_out, C5_SIZE, F6_SIZE
    );
    relu_forward<<<(F6_SIZE + 255) / 256, 256>>>(model->d_fc2_out, F6_SIZE);
    
    // FC3: 84 -> 10
    fc_forward<<<(OUTPUT_SIZE + 255) / 256, 256>>>(
        model->d_fc2_out, model->d_fc3_w, model->d_fc3_b,
        model->d_output, F6_SIZE, OUTPUT_SIZE
    );
    
    // Softmax
    softmax_forward<<<1, OUTPUT_SIZE>>>(model->d_output, OUTPUT_SIZE);
    
    CHECK_CUDA(cudaGetLastError());
}

void backward_pass(LeNet5* model, int label, float lr) {
    // Compute gradient at output (cross-entropy loss derivative)
    float h_output[OUTPUT_SIZE];
    CHECK_CUDA(cudaMemcpy(h_output, model->d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        h_output[i] = (i == label) ? h_output[i] - 1.0f : h_output[i];
    }
    CHECK_CUDA(cudaMemcpy(model->d_grad_output, h_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // ===== Backprop FC3 (84 -> 10) =====
    fc_backward_weight<<<OUTPUT_SIZE, F6_SIZE>>>(
        model->d_fc2_out, model->d_grad_output, model->d_fc3_w, F6_SIZE, OUTPUT_SIZE, lr
    );
    fc_backward_bias<<<1, OUTPUT_SIZE>>>(model->d_grad_output, model->d_fc3_b, OUTPUT_SIZE, lr);
    fc_backward_input<<<(F6_SIZE + 255) / 256, 256>>>(
        model->d_grad_output, model->d_fc3_w, model->d_grad_fc2, F6_SIZE, OUTPUT_SIZE
    );
    
    // Backprop through ReLU
    relu_backward<<<(F6_SIZE + 255) / 256, 256>>>(model->d_grad_fc2, model->d_fc2_out, F6_SIZE);
    
    // ===== Backprop FC2 (120 -> 84) =====
    fc_backward_weight<<<F6_SIZE, C5_SIZE>>>(
        model->d_fc1_out, model->d_grad_fc2, model->d_fc2_w, C5_SIZE, F6_SIZE, lr
    );
    fc_backward_bias<<<1, F6_SIZE>>>(model->d_grad_fc2, model->d_fc2_b, F6_SIZE, lr);
    fc_backward_input<<<(C5_SIZE + 255) / 256, 256>>>(
        model->d_grad_fc2, model->d_fc2_w, model->d_grad_fc1, C5_SIZE, F6_SIZE
    );
    
    // Backprop through ReLU
    relu_backward<<<(C5_SIZE + 255) / 256, 256>>>(model->d_grad_fc1, model->d_fc1_out, C5_SIZE);
    
    // ===== Backprop FC1 (256 -> 120) =====
    fc_backward_weight<<<C5_SIZE, C3_FILTERS * S4_SIZE * S4_SIZE>>>(
        model->d_pool2_out, model->d_grad_fc1, model->d_fc1_w,
        C3_FILTERS * S4_SIZE * S4_SIZE, C5_SIZE, lr
    );
    fc_backward_bias<<<1, C5_SIZE>>>(model->d_grad_fc1, model->d_fc1_b, C5_SIZE, lr);
    fc_backward_input<<<(C3_FILTERS * S4_SIZE * S4_SIZE + 255) / 256, 256>>>(
        model->d_grad_fc1, model->d_fc1_w, model->d_grad_pool2,
        C3_FILTERS * S4_SIZE * S4_SIZE, C5_SIZE
    );
    
    // ===== Backprop Pool2 (8x8x16 -> 4x4x16) =====
    CHECK_CUDA(cudaMemset(model->d_grad_conv2, 0, C3_FILTERS * C3_SIZE * C3_SIZE * sizeof(float)));
    dim3 pool2_grid(C3_FILTERS, S4_SIZE, 1);
    maxpool2d_backward<<<pool2_grid, S4_SIZE>>>(
        model->d_grad_pool2, model->d_pool2_indices, model->d_grad_conv2,
        C3_FILTERS, C3_SIZE, S4_SIZE
    );
    
    // Backprop through ReLU
    relu_backward<<<(C3_FILTERS * C3_SIZE * C3_SIZE + 255) / 256, 256>>>(
        model->d_grad_conv2, model->d_conv2_out, C3_FILTERS * C3_SIZE * C3_SIZE
    );
    
    // ===== Backprop Conv2 (12x12x6 -> 8x8x16) =====
    dim3 conv2_w_grid(C3_FILTERS, C1_FILTERS, CONV_KERNEL);
    conv2d_backward_weight<<<conv2_w_grid, CONV_KERNEL>>>(
        model->d_pool1_out, model->d_grad_conv2, model->d_conv2_w,
        C1_FILTERS, C3_FILTERS, S2_SIZE, C3_SIZE, CONV_KERNEL, lr
    );
    conv2d_backward_bias<<<1, C3_FILTERS>>>(
        model->d_grad_conv2, model->d_conv2_b, C3_FILTERS, C3_SIZE, lr
    );
    dim3 conv2_in_grid(C1_FILTERS, S2_SIZE, 1);
    conv2d_backward_input<<<conv2_in_grid, S2_SIZE>>>(
        model->d_grad_conv2, model->d_conv2_w, model->d_grad_pool1,
        C1_FILTERS, C3_FILTERS, S2_SIZE, C3_SIZE, CONV_KERNEL
    );
    
    // ===== Backprop Pool1 (24x24x6 -> 12x12x6) =====
    CHECK_CUDA(cudaMemset(model->d_grad_conv1, 0, C1_FILTERS * C1_SIZE * C1_SIZE * sizeof(float)));
    dim3 pool1_grid(C1_FILTERS, S2_SIZE, 1);
    maxpool2d_backward<<<pool1_grid, S2_SIZE>>>(
        model->d_grad_pool1, model->d_pool1_indices, model->d_grad_conv1,
        C1_FILTERS, C1_SIZE, S2_SIZE
    );
    
    // Backprop through ReLU
    relu_backward<<<(C1_FILTERS * C1_SIZE * C1_SIZE + 255) / 256, 256>>>(
        model->d_grad_conv1, model->d_conv1_out, C1_FILTERS * C1_SIZE * C1_SIZE
    );
    
    // ===== Backprop Conv1 (28x28x1 -> 24x24x6) =====
    dim3 conv1_w_grid(C1_FILTERS, INPUT_CHANNELS, CONV_KERNEL);
    conv2d_backward_weight<<<conv1_w_grid, CONV_KERNEL>>>(
        model->d_input, model->d_grad_conv1, model->d_conv1_w,
        INPUT_CHANNELS, C1_FILTERS, INPUT_SIZE, C1_SIZE, CONV_KERNEL, lr
    );
    conv2d_backward_bias<<<1, C1_FILTERS>>>(
        model->d_grad_conv1, model->d_conv1_b, C1_FILTERS, C1_SIZE, lr
    );
    
    CHECK_CUDA(cudaGetLastError());
}

void free_lenet5(LeNet5* model) {
    cudaFree(model->d_input);
    cudaFree(model->d_conv1_out); cudaFree(model->d_pool1_out);
    cudaFree(model->d_conv2_out); cudaFree(model->d_pool2_out);
    cudaFree(model->d_fc1_out); cudaFree(model->d_fc2_out); cudaFree(model->d_output);
    cudaFree(model->d_conv1_w); cudaFree(model->d_conv1_b);
    cudaFree(model->d_conv2_w); cudaFree(model->d_conv2_b);
    cudaFree(model->d_fc1_w); cudaFree(model->d_fc1_b);
    cudaFree(model->d_fc2_w); cudaFree(model->d_fc2_b);
    cudaFree(model->d_fc3_w); cudaFree(model->d_fc3_b);
    cudaFree(model->d_grad_output); cudaFree(model->d_grad_fc2); cudaFree(model->d_grad_fc1);
    cudaFree(model->d_grad_pool2); cudaFree(model->d_grad_conv2);
    cudaFree(model->d_grad_pool1); cudaFree(model->d_grad_conv1);
    cudaFree(model->d_pool1_indices); cudaFree(model->d_pool2_indices);
}

// ==================== Training and Testing ====================

void train_epoch(LeNet5* model, float* images, unsigned char* labels, int num_samples, float lr) {
    for (int i = 0; i < num_samples; i++) {
        CHECK_CUDA(cudaMemcpy(model->d_input, &images[i * INPUT_SIZE * INPUT_SIZE],
                             INPUT_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        forward_pass(model);

        backward_pass(model, labels[i], lr);
        
        if ((i + 1) % 1000 == 0) {
            printf("  Processed %d/%d samples\r", i + 1, num_samples);
            fflush(stdout);
        }
    }
    printf("\n");
}

float test_accuracy(LeNet5* model, float* images, unsigned char* labels, int num_samples) {
    int correct = 0;
    float h_output[OUTPUT_SIZE];
    
    for (int i = 0; i < num_samples; i++) {
        CHECK_CUDA(cudaMemcpy(model->d_input, &images[i * INPUT_SIZE * INPUT_SIZE],
                             INPUT_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        
        forward_pass(model);
        
        CHECK_CUDA(cudaMemcpy(h_output, model->d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        
        int predicted = 0;
        float max_prob = h_output[0];
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (h_output[j] > max_prob) {
                max_prob = h_output[j];
                predicted = j;
            }
        }
        
        if (predicted == labels[i]) correct++;
    }
    
    return 100.0f * correct / num_samples;
}

// ==================== Main ====================

int main(int argc, char** argv) {
    printf("LeNet-5 MNIST Training\n");
    printf("=============================================\n\n");
    
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <train-images> <train-labels> <test-images> <test-labels>\n", argv[0]);
        fprintf(stderr, "Example: %s train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte\n", argv[0]);
        return 1;
    }

    printf("Loading MNIST dataset...\n");
    MNISTData train_data = load_mnist_train(argv[1], argv[2]);
    MNISTData test_data = load_mnist_test(argv[3], argv[4]);
    
    printf("Loaded %d training samples and %d test samples\n\n", train_data.count, test_data.count);
    
    printf("Initializing LeNet-5 model...\n");
    LeNet5 model;
    init_lenet5(&model);
    printf("Model initialized\n\n");

    printf("Starting training...\n");
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        printf("Epoch %d/%d\n", epoch + 1, EPOCHS);
        
        train_epoch(&model, train_data.images, train_data.labels, train_data.count, LEARNING_RATE);
        
        float train_acc = test_accuracy(&model, train_data.images, train_data.labels, 1000); // Sample
        float test_acc = test_accuracy(&model, test_data.images, test_data.labels, test_data.count);
        
        printf("  Train Accuracy: %.2f%% | Test Accuracy: %.2f%%\n\n", train_acc, test_acc);
    }
    
    printf("Training complete!\n");
    printf("Final Test Accuracy: %.2f%%\n", test_accuracy(&model, test_data.images, test_data.labels, test_data.count));
    
    free(train_data.images);
    free(train_data.labels);
    free(test_data.images);
    free(test_data.labels);
    free_lenet5(&model);
    
    return 0;
}
