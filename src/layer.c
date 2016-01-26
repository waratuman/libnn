#include <stdlib.h>

#include "layer.h"
#include "layers/lrn.h"
#include "layers/convolutional.h"
#include "layers/fully_connected.h"
#include "layers/singly_connected.h"

void nn_layer_activate(LAYER_TYPE type, void* layer, float* input, float* output)
{
    switch(type) {
        case LRN:
        nn_layer_activate_lrn(layer, input, output);
        break;

        case CONVOLUTIONAL:
        nn_layer_activate_convolutional(layer, input, output);
        break;

        case FULLY_CONNECTED:
        nn_layer_activate_fully_connected(layer, input, output);
        break;

        case SINGLY_CONNECTED:
        nn_layer_activate_singly_connected(layer, input, output);
        break;
    }
}

void nn_layer_destroy(LAYER_TYPE type, void* layer)
{
    switch(type) {
        case LRN:
        nn_layer_destroy_lrn(layer);
        break;

        case CONVOLUTIONAL:
        nn_layer_destroy_convolutional(layer);
        break;

        case FULLY_CONNECTED:
        nn_layer_destroy_fully_connected(layer);
        break;

        case SINGLY_CONNECTED:
        nn_layer_destroy_singly_connected(layer);
        break;
    }
}

int nn_layer_input_count(LAYER_TYPE type, void* layer)
{
    switch(type) {
        case LRN:
        return ((nn_layer_lrn_t*)layer)->inputCount;
        break;

        case CONVOLUTIONAL:
        return ((nn_layer_convolutional_t*)layer)->inputCount;
        break;

        case FULLY_CONNECTED:
        return ((nn_layer_fully_connected_t*)layer)->inputCount;
        break;

        case SINGLY_CONNECTED:
        return ((nn_layer_singly_connected_t*)layer)->inputCount;
        break;
    }

    return 0;
}

int nn_layer_output_count(LAYER_TYPE type, void* layer)
{
    switch(type) {
        case LRN:
        return ((nn_layer_lrn_t*)layer)->outputCount;
        break;

        case CONVOLUTIONAL:
        return ((nn_layer_convolutional_t*)layer)->outputCount;
        break;

        case FULLY_CONNECTED:
        return ((nn_layer_fully_connected_t*)layer)->outputCount;
        break;

        case SINGLY_CONNECTED:
        return ((nn_layer_singly_connected_t*)layer)->outputCount;
        break;
    }

    return 0;
}

int nn_layer_input_dimension_count(LAYER_TYPE type, void* layer)
{
    switch(type) {
        case LRN:
        return ((nn_layer_lrn_t*)layer)->inputDimensionCount;
        break;

        case CONVOLUTIONAL:
        return ((nn_layer_convolutional_t*)layer)->inputDimensionCount;
        break;

        case FULLY_CONNECTED:
        return 1;
        break;

        case SINGLY_CONNECTED:
        return 1;
        break;
    }
}

int nn_layer_output_dimension_count(LAYER_TYPE type, void* layer)
{
    switch(type) {
        case LRN:
        return ((nn_layer_lrn_t*)layer)->outputDimensionCount;
        break;

        case CONVOLUTIONAL:
        return ((nn_layer_convolutional_t*)layer)->outputDimensionCount;
        break;

        case FULLY_CONNECTED:
        return 0;
        break;

        case SINGLY_CONNECTED:
        return 0;
        break;
    }
}

int* nn_layer_input_dimensions(LAYER_TYPE type, void* layer)
{
    switch(type) {
        case LRN:
        return ((nn_layer_lrn_t*)layer)->inputDimensions;
        break;

        case CONVOLUTIONAL:
        return ((nn_layer_convolutional_t*)layer)->inputDimensions;
        break;

        case FULLY_CONNECTED:
        return NULL;//&((nn_layer_fully_connected_t*)layer)->inputCount;
        break;

        case SINGLY_CONNECTED:
        return NULL; //&((nn_layer_singly_connected_t*)layer)->inputCount;
        break;
    }
}

int* nn_layer_output_dimensions(LAYER_TYPE type, void* layer)
{
    switch(type) {
        case LRN:
        return ((nn_layer_lrn_t*)layer)->outputDimensions;
        break;

        case CONVOLUTIONAL:
        return ((nn_layer_convolutional_t*)layer)->outputDimensions;
        break;

        case FULLY_CONNECTED:
        return NULL; //&((nn_layer_fully_connected_t*)layer)->outputCount;
        break;

        case SINGLY_CONNECTED:
        return NULL; //&((nn_layer_singly_connected_t*)layer)->outputCount;
        break;
    }
}