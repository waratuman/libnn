#pragma once

#include "layer.h"
#include "layers/singly_connected.h"
#include "layers/fully_connected.h"
#include "layers/convolutional.h"

#include "errors.h"

typedef struct nn_network_t {
    int layerCount;
    LAYER_TYPE* layerTypes;
    void** layers;

    nn_error_fn error;      // Error / Loss function (defaults to the sum of squares)

    float** activations;    // The stored node activations (for backpropagation)
    float** derivatives;    // The stored node activation derivatives (for backpropagation)
} nn_network_t;

void nn_network_init(nn_network_t *network);

nn_network_t* nn_network_create(
    int layerCount,
    LAYER_TYPE* layerTypes,
    void** layers
);

// Note: Will also call destroy on any layers
void nn_network_destroy(nn_network_t* network);

// Input is the size of the first layer in the network
// Output is of the size of the last layer in the network
void nn_network_activate(nn_network_t *network, float* input, float* output);

float nn_network_loss(nn_network_t* network, float* input, float* target);

void nn_network_train(nn_network_t* network, float* input, float* target);
