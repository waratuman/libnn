/*
Fully connected layers connect every input to every output using a integration
function. For each output a given activation function is applied. The integration
funciton is of the form:

    typedef float (*nn_integration_fn)(int count, float* a, float* b);

And the activation function is of the form:

    typedef float (*nn_activation_fn)(float x);

For each output i the result is:

    o_i &= a \left( g\left( \hat{w_i}, \hat{\imath} \right) + b_i \right)

Where a is the activation function and g is the integration function. In a typical
fully connected layer the integration function is the sum of products:

    o_i &= a \left( g\left( \hat{w_i}, \hat{\imath} \right) + b_i \right)\\
    g\left( \hat{w_i}, \hat{\imath} \right) &= \sum_{j=0}^{n} \left( w_{ij}i_j \right)

Where i is the ith output, j is the jth input, n is the number of
inputs, w_ij is the weight associated between the ith ouput and jth
input, i_j is the jth input and b_i is the bias of the ith output.
*/
#pragma once

#include "layer.h"
#include "activations.h"
#include "integrations.h"

typedef struct {
    int inputCount;                     // Number of inputs to the layer
    int outputCount;                    // Number of outputs of the layer
    nn_activation_fn activation;        // Activation function
    nn_integration_fn integration;      // Integration function

    float* biases;
    float** weights;
} nn_layer_fully_connected_t;

void nn_layer_init_fully_connected(nn_layer_fully_connected_t *layer);

nn_layer_fully_connected_t* nn_layer_create_fully_connected(nn_activation_fn activation, nn_integration_fn integration, int inputCount, int outputCount);

void nn_layer_destroy_fully_connected(nn_layer_fully_connected_t* layer);

void nn_layer_integrate_fully_connected(nn_layer_fully_connected_t *layer, float* input, float* output);

void nn_layer_activate_fully_connected(nn_layer_fully_connected_t *layer, float* input, float* output);



// #include "layers/convolutional.h"
//
// typedef nn_layer_convolutional_t nn_layer_fully_connected_t;
//
// nn_layer_fully_connected_t* nn_layer_create_fully_connected(nn_activation_fn a, nn_integration_fn i, int inputCount, int outputCount);
//
// void nn_layer_destroy_fully_connected(nn_layer_fully_connected_t* layer);
//
// void nn_layer_integrate_fully_connected(nn_layer_fully_connected_t *layer, float* input, float* output);
//
// void nn_layer_activate_fully_connected(nn_layer_fully_connected_t *layer, float* input, float* output);
