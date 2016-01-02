#include <stdbool.h>
#include "layer.h"

typedef struct {
    int inputCount;         // Number of inputs
    int outputCount;        // Number of outputs
    int weightCount;        // Number of weights
    int dimensionCount;     // Number of dimensions
    int kernelCount;        // Number of kernels

    nn_activation_fn activation;    // Activation function
    nn_integration_fn integration;  // Integration function

    float bias;              // The bias of the kernel
    float* weights;          // The weights of the kernel

    int* size;               // The size of the kernel in each dimension
    int* stride;             // The stride in each dimension, defaults to 1
    int* padding;            // The padding in each dimension, defaults to 0
    int* inputDimensions;    // The dimensions of the input
    int* outputDimensions;   // The dimensions of the output
} nn_layer_convolutional_t;

void nn_layer_init_convolutional(nn_layer_convolutional_t *layer);

nn_layer_convolutional_t* nn_layer_create_convolutional(
    nn_activation_fn activation,
    int inputCount,
    int dimensionCount,
    int kernelCount,
    int* dimensions,
    int* padding,
    int* kernel_stride,
    int* kernel_size
);

void nn_layer_destroy_convolutional(
    nn_layer_convolutional_t* layer
);

void nn_layer_activate_convolutional(
    nn_layer_convolutional_t *layer,
    float* input,
    float* output
);

// Returns true if the given index (an array of the size layer->dimensionCount)
// lies withing the padded region (outside the input). False if it lies within
// the input.
bool nn_layer_convolutional_is_index_padding(
    nn_layer_convolutional_t *layer,
    int* index
);


















// #include "kernel.h"
// // Index mapping for the kernel.
// // Since inputs are a 1 dimensional array mapping is necessary if you
// // have a 2d or 3d image that you would like the kernel to operate on.
// // For example and image with width 10 and height 10 would have an input sized
// // 100 and the kernel with size 2 x 2 would have a mapping:
// //
// //   k1 = i1, k2 = i2, k3 = i11, k4 = i12
// //   int mapping(int output_index, int kernel_index, int input_size, int kernel_size)
// //   {
// //      return (i / sqrt(k->size) * sqrt(l->inputs)) + i % sqrt(k->size);
// //   }
// // typedef int (*nn_kernel_mapping_fn)(nn_layer_convolutional_t* l, layer_size, nn_kernel_t k, int i);
//
// typedef int (*nn_layer_mapping_fn)(int output_index, int kernel_index, int input_size, int kernel_size);
//
// /* Convolutional layers apply a kernel to the input and produce a resulting output.
//
// The total number of outputs is equal to:
//
//     \frac{i_w - k_w + 2p_w}{s_w} + 1
//
// Where:
//
//     i_w = Number of inputs
//     k_w = Kernel width
//     p_w = Kernel padding
//     s_w = Kernel stride
//
// */
// typedef struct {
//     int inputs;              // Number of inputs
//     int outputs;             // Number of outputs
//
//     int stride;              // Kernel stride
//     int padding;             // Kernel padding
//
//     float *weights;          // Kernel weights
//     int kernel_size;         // Kernel size
//
//     nn_activation_fn activation;    // Activation function
//     nn_layer_mapping_fn mapping;    // Input to kernel mapping function
// } nn_layer_convolutional_t;
//
// void nn_layer_init_convolutional(nn_layer_convolutional_t *layer);
// nn_layer_convolutional_t* nn_layer_create_convolutional(nn_activation_fn fn, int inputs, int kernel_size, int kernel_stride, int kernel_padding);
// void nn_layer_destroy_convolutional(nn_layer_convolutional_t* layer);
// void nn_layer_activate_convolutional(nn_layer_convolutional_t *layer, float* input, float* output);
