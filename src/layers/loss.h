#pragma once

typedef struct {
    int inputCount;                 // Number of inputs
    nn_activation_fn activation;    // Activation function
} nn_layer_loss_t;

void nn_layer_init_loss(nn_layer_loss_t *layer);

nn_layer_loss_t* nn_layer_create_loss(nn_activation_fn activation, int inputCount);

void nn_layer_destroy_loss(nn_layer_loss_t* layer);

// output: The targeted output (expected output)
void nn_layer_integrate_loss(nn_layer_loss_t *layer, float* input, float* output);

// output: The targeted output (expected output)
float nn_layer_activate_loss(nn_layer_loss_t *layer, float* input, float* output);
