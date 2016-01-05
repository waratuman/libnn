// Ref: http://page.mi.fu-berlin.de/rojas/neural/
#ifndef LAYER_H
#define LAYER_H

#include <stdbool.h>

#include "util.h"
#include "activations.h"
#include "integrations.h"

typedef struct {
    int inputCount;             // Number of inputs
    int outputCount;            // Number of outputs
    int weightCount;            // Number of weights
    int kernelCount;            // Number of kernels
    int inputDimensionCount;    // Number of input dimensions
    int outputDimensionCount;   // Number of output dimensions

    nn_activation_fn activation;    // Activation function
    nn_integration_fn integration;  // Integration function

    float* biases;           // The bias of the kernel
    float** weights;         // The weights of the kernel

    int* size;               // The size of the kernel in each dimension
    int* stride;             // The stride in each dimension, defaults to 1
    int* padding;            // The padding in each dimension, defaults to 0
    int* inputDimensions;    // The dimensions of the input
    int* outputDimensions;   // The dimensions of the output
} nn_layer_t;

void nn_layer_init(nn_layer_t *layer);

nn_layer_t* nn_layer_create(
    nn_activation_fn activation,
    nn_integration_fn integration,
    int inputCount,
    int inputDimensionCount,
    int kernelCount,
    int* dimensions,
    int* padding,
    int* kernel_stride,
    int* kernel_size
);
    
nn_layer_t* nn_layer_create_connected(
    nn_activation_fn activation,
    nn_integration_fn integration,
    int inputCount,
    int outputCount
);

void nn_layer_destroy(
    nn_layer_t* layer
);

// The output has an extra dimension (first dimension, index 0),
// which is the number for kernels.
void nn_layer_activate(
    nn_layer_t *layer,
    float* input,
    float* output
);

// Returns true if the given index (an array of the size layer->inputDimensionCount)
// lies withing the padded region (outside the input). False if it lies within
// the input.
bool nn_layer_is_index_padding(
    nn_layer_t *layer,
    int* index
);

#endif