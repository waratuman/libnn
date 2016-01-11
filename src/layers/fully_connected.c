#include <stdlib.h>

// #include "activations.h"
#include "layer.h"
#include "layers/fully_connected.h"

void nn_layer_init_fully_connected(nn_layer_fully_connected_t *l)
{
    l->biases = calloc(l->outputCount, sizeof(float));
    l->weights = calloc(l->outputCount, sizeof(float*));

    for (int i = 0; i < l->outputCount; i++) {
        l->weights[i] = calloc(l->inputCount, sizeof(float));
    }
}

nn_layer_fully_connected_t* nn_layer_create_fully_connected(nn_activation_fn a, nn_integration_fn i, int ic, int oc)
{
    nn_layer_fully_connected_t* l = calloc(1, sizeof(nn_layer_fully_connected_t));
    l->activation = a;
    l->integration = i;
    l->inputCount = ic;
    l->outputCount = oc;
    nn_layer_init_fully_connected(l);
    return l;
}

void nn_layer_destroy_fully_connected(nn_layer_fully_connected_t* l)
{
    for (int i = 0; i < l->outputCount; i++) {
        free(l->weights[i]);
    }
    free(l->weights);
    free(l->biases);
    free(l);
}

void nn_layer_integrate_fully_connected(nn_layer_fully_connected_t *l, float* input, float* output)
{

    float* integrationInputs[2] = {input, NULL};
    for (int i = 0; i < l->outputCount; i++) {
        integrationInputs[1] = l->weights[i];
        output[i] = l->integration(l->inputCount, integrationInputs) + l->biases[i];
    }
}

void nn_layer_activate_fully_connected(nn_layer_fully_connected_t *l, float* input, float* output)
{
    nn_layer_integrate_fully_connected(l, input, output);
    for (int i = 0; i < l->outputCount; i++) {
        output[i] = l->activation(&output[i], 0);
    }
}
