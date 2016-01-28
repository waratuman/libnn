#include <nn.h>
#include <stdlib.h>

void nn_layer_init_singly_connected(nn_layer_singly_connected_t *l)
{
    l->outputCount = l->inputCount;
    l->weightCount = l->inputCount;

    l->biases = calloc(l->outputCount, sizeof(float));
    l->weights = calloc(l->weightCount, sizeof(float));
}

nn_layer_singly_connected_t* nn_layer_create_singly_connected(nn_activation_fn a, int ic)
{
    nn_layer_singly_connected_t* l = calloc(1, sizeof(nn_layer_singly_connected_t));
    l->activation = a;
    l->inputCount = ic;

    nn_layer_init_singly_connected(l);
    return l;
}

void nn_layer_destroy_singly_connected(nn_layer_singly_connected_t* l)
{
    free(l->biases);
    free(l->weights);
    free(l);
}

void nn_layer_aggregate_singly_connected(nn_layer_singly_connected_t *l, float* input, float* output)
{
    for (int i = 0; i < l->inputCount; i++) {
        output[i] = input[i] + l->biases[i];
    }
    
}

void nn_layer_activate_singly_connected(nn_layer_singly_connected_t *l, float* input, float* output)
{
    nn_layer_aggregate_singly_connected(l, input, output);
    for (int i = 0; i < l->inputCount; i++) {
        output[i] = l->activation(&output[i], 0);
    }
}
