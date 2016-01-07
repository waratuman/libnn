#include <stdlib.h>

#include "network.h"

void nn_network_init(nn_network_t *n)
{
    if (n->error == NULL) {
        n->error = squared_error;
    }
}

nn_network_t* nn_network_create(int lc, LAYER_TYPE* layerTypes, void** layers)
{
    nn_network_t* n = calloc(1, sizeof(nn_network_t));

    n->layerCount = lc;
    n->layerTypes = layerTypes;
    n->layers = layers;

    nn_network_init(n);
    return n;
}

// Note: Will also call destroy on any layers
void nn_network_destroy(nn_network_t* n)
{
    for (int i = 0; i < n->layerCount; i++) {
        nn_layer_destroy(n->layerTypes[i], n->layers[i]);
    }
    free(n);
}

void nn_network_activate(nn_network_t *n, float* input, float* output)
{
    float* layerInput = input;
    float* layerOutput = NULL;

    for (int i = 0; i < n->layerCount - 1; i++) {
        layerOutput = calloc(nn_layer_output_count(n->layerTypes[i], n->layers[i]), sizeof(float));
        nn_layer_activate(n->layerTypes[i], n->layers[i], layerInput, layerOutput);

        if (i > 0) {
            free(layerInput);
        }
        layerInput = layerOutput;
    }

    nn_layer_activate(n->layerTypes[n->layerCount - 1], n->layers[n->layerCount - 1], layerInput, output);
    free(layerOutput);
}

float nn_network_loss(nn_network_t* n, float* input, float* target)
{
    int lossInputCount = nn_layer_output_count(n->layerTypes[n->layerCount - 1], n->layers[n->layerCount - 1]);
    float* lossInput = calloc(lossInputCount, sizeof(float));
    nn_network_activate(n, input, lossInput);

    float error;
    for (int i = 0; i < lossInputCount; i++) {
        error = error + n->error(lossInput[i], target[i], 0);
    }
    error = error / lossInputCount;

    free(lossInput);

    return error;
}

// Train the network using backpropagation
void nn_network_train(nn_network_t* n, float* inputs, float* target)
{
    // n->activations = calloc(n->layerCount, sizeof(float*));
    // n->derivatives = calloc(n->layerCount, sizeof(float*));
    // for (int i = 0; i < n->layerCount; i++) {
    //     n->activations[i] = calloc(n->layers[i]->outputCount, sizeof(float));
    //     n->derivatives[i] = calloc(n->layers[i]->outputCount, sizeof(float));
    // }
    //
    // // Step 1: Activate the network, storing the node activations and derivatives (including loss layer)
    // for (int i = 0; i < n->layerCount; i++) {
    //     nn_layer_integrate(n->layers[i], inputs, n->activations[i]);
    //     for (int j = 0; j < n->layers[i]->outputCount; j++) {
    //         n->derivatives[i][j] = n->layers[i]->activation(n->activations[i][j], 0);
    //         n->activations[i][j] = n->layers[i]->activation(n->activations[i][j], 0);
    //     }
    // }
    //
    // // Step 2: Backpropagate the network with an input of 1 fed into the output
    // //         multipling by the derivative.
    //
    // for (int i = 0; i < n->layerCount; i++) {
    //     free(n->activations[i]);
    //     free(n->derivatives[i]);
    // }
    // free(n->activations);
    // free(n->derivatives);
    //
}
