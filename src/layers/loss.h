#include <stdbool.h>
#include "layer.h"

typedef struct {
    int inputCount;                 // Number of inputs
    nn_integration_fn fn;      // Loss function
} nn_layer_loss_t;

void nn_layer_init_loss(nn_layer_loss_t *layer);

nn_layer_loss_t* nn_layer_create_loss(
    nn_integration_fn loss_fn,
    int inputCount
);
    
void nn_layer_destroy_loss(
    nn_layer_loss_t* layer
);

// The output has an extra dimension (first dimension, index 0),
// which is the number for kernels.
float nn_layer_activate_loss(
    nn_layer_loss_t *layer,
    float* input,
    float* output
);
