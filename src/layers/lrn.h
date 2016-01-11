// http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
#pragma once

#include <stdbool.h>

#include "util.h"
#include "activations.h"
#include "integrations.h"

// b_{x,y}^i = \frac{a_{x,y}^i}
// {
//   \left(
//     k + \alpha \sum_{j=\max(0, i-n/2)}^{\min(N-1,i+n/2)}\left(a_{x,y}^j\right)^2
//   \right)^\beta
// }
typedef struct {
    int inputCount;             // Number of inputs
    int outputCount;            // Number of outputs (== inputCount)
    int kernelInputCount;       // Number of inputs to the kernel
    int inputDimensionCount;    // Number of input dimensions
    int outputDimensionCount;   // Number of output dimensions

    nn_activation_fn activation;    // Activation function
    nn_integration_fn integration;  // Integration function

    float k;                // defaults to 2
    float alpha;            // defaults to 10^-4
    float beta;             // defaults to 0.75

    int* size;              // The size of the kernel in each dimension
    int* inputDimensions;   // The dimensions of the input
    int* outputDimensions;  // The dimensions of the output (== inputDimesions)
} nn_layer_lrn_t;

void nn_layer_init_lrn(nn_layer_lrn_t *layer);

nn_layer_lrn_t* nn_layer_create_lrn(
    int inputCount,
    int inputDimensionCount,
    int* dimensions,
    int* kernel_size,
    float k,
    float alpha,
    float beta
);

void nn_layer_destroy_lrn(
    nn_layer_lrn_t* layer
);

// The output has an extra dimension (first dimension, index 0),
// which is the number for kernels.
void nn_layer_integrate_lrn(
    nn_layer_lrn_t *l,
    float* input,
    float* output
);

// The output has an extra dimension (first dimension, index 0),
// which is the number for kernels.
void nn_layer_activate_lrn(
    nn_layer_lrn_t *layer,
    float* input,
    float* output
);

// Returns true if the given index (an array of the size layer->inputDimensionCount)
// lies withing the padded region (outside the input). False if it lies within
// the input.
bool nn_layer_is_lrn_index_padding(
    nn_layer_lrn_t *layer,
    int* index
);
