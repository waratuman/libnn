#pragma once

#include "layer.h"
#include "activations.h"

typedef struct {
    int inputCount;             // Number of inputs
    int outputCount;            // Number of outputs
    int weightCount;            // Number of weights

    nn_activation_fn activation;    // Activation function

    float* biases;           // The bias of the kernel
    float* weights;          // The weights of the kernel
} nn_layer_singly_connected_t;

void nn_layer_init_singly_connected(nn_layer_singly_connected_t *layer);

nn_layer_singly_connected_t* nn_layer_create_singly_connected(nn_activation_fn activation, int inputCount);

void nn_layer_destroy_singly_connected(nn_layer_singly_connected_t* layer);

void nn_layer_integrate_singly_connected(nn_layer_singly_connected_t *layer, float* input, float* output);

void nn_layer_activate_singly_connected(nn_layer_singly_connected_t *layer, float* input, float* output);
