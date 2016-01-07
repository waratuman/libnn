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

    for (int i = 0; i < l->outputCount; i++) {
        output[i] = l->integration(l->inputCount, l->weights[i], input) + l->biases[i];
    }
}

void nn_layer_activate_fully_connected(nn_layer_fully_connected_t *l, float* input, float* output)
{
    nn_layer_integrate_fully_connected(l, input, output);
    for (int i = 0; i < l->outputCount; i++) {
        output[i] = l->activation(output[i], 0);
    }
}
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
// nn_layer_fully_connected_t* nn_layer_create_fully_connected(nn_activation_fn a, nn_integration_fn i, int ic, int oc)
// {
//     int dim[1] = {ic};
//     int pad[1] = {0};
//     int str[1] = {1};
//     int siz[1] = {ic};
//
//     nn_layer_fully_connected_t* l = (nn_layer_fully_connected_t*)nn_layer_create_convolutional(a, i, ic, 1, oc, dim, pad, str, siz);
//     return l;
// }
//
// void nn_layer_destroy_fully_connected(nn_layer_fully_connected_t* l)
// {
//     nn_layer_destroy_fully_connected((nn_layer_convolutional_t*)l);
// }
//
// void nn_layer_integrate_fully_connected(nn_layer_fully_connected_t *l, float* input, float* output)
// {
//     nn_layer_integrate_convolutional((nn_layer_convolutional_t*)l, input, output);
// }
//
// void nn_layer_activate_fully_connected(nn_layer_fully_connected_t *l, float* input, float* output)
// {
//     nn_layer_activate_convolutional((nn_layer_convolutional_t*)l, input, output);
// }
//
