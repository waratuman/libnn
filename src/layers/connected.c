#include <stdlib.h>

#include "connected.h"

#include "utils.h"

void nn_layer_init_connected(nn_layer_connected_t *l)
{
    l->biases = calloc(l->outputCount, sizeof(float));
    l->weights = calloc(l->outputCount, sizeof(float*));

    for (int i = 0; i < l->outputCount; i++) {
        l->weights[i] = calloc(l->inputCount, sizeof(float));
    }

    if (l->integration == NULL) {
        l->integration = sum_of_products_integration;
    }
}

nn_layer_connected_t* nn_layer_create_connected(nn_activation_fn fn, int i, int o)
{
    nn_layer_connected_t* l = calloc(1, sizeof(nn_layer_connected_t));
    l->activation = fn;
    l->inputCount = i;
    l->outputCount = o;
    nn_layer_init_connected(l);
    return l;
}

void nn_layer_destroy_connected(nn_layer_connected_t* l)
{
    free(l->biases);

    for (int i = 0; i < l->outputCount; i++) {
        free(l->weights[i]);
    }

    free(l->weights);
    free(l);
}

void nn_layer_activate_connected(nn_layer_connected_t *l, float* input, float* output)
{
    for (int i = 0; i < l->outputCount; i++) {
        output[i] = l->activation(l->integration(l->inputCount, l->weights[i], input) + l->biases[i]);
    }
}