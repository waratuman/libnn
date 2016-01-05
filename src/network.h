#include "layer.h"

typedef struct {
    int layerCount;

    nn_layer_t* layers;

    float** activations;      // The stored node activations (for backpropagation)
    float** derivatives;      // The stored node activation derivatives (for backpropagation)
} nn_network_t;

void nn_network_init(nn_network_t *network);

nn_network_t* nn_network_create(
    int layerCount,
    nn_layer_t* layers
);
    
void nn_network_destroy(
    nn_network_t* network
);

// Input is the size of the first layer in the network
// Output is of the size of the last layer in the network
void nn_network_activate(
    nn_network_t *network,
    float* input,
    float* output
);
