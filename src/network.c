#include <stdlib.h>

#include "network.h"

void nn_network_init(nn_network_t *network)
{
    
}

nn_network_t* nn_network_create(int layerCount, nn_layer_t* layers)
{
    nn_network_t* n = calloc(1, sizeof(nn_network_t));

    n->layerCount = layerCount;
    n->layers = layers;

    nn_network_init(n);

    return n;
}

void nn_network_destroy(nn_network_t* n)
{
    free(n);
}

void nn_network_activate(nn_network_t *n, float* input, float* output)
{
    // TODO: Don't need all of this allocation, can just allocate the largest
    // input / ouput of any layer and keep reusing
    float** layerInputs = calloc(n->layerCount, sizeof(float*));
    layerInputs[0] = input;
    
    for (int i = 0; i < n->layerCount - 1; i++) {
        layerInputs[i + 1] = calloc(n->layers[i + 1].inputCount, sizeof(float));
        nn_layer_activate(&n->layers[i], layerInputs[i], layerInputs[i + 1]);
    }

    nn_layer_activate(&n->layers[n->layerCount - 1], layerInputs[n->layerCount - 1], output);
}

