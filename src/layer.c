#include <nn.h>
#include <stdlib.h>



void nn_layer_activate(nn_layer_type_t type, void* layer, float* input, float* output)
{
    switch(type) {
        case NN_LRN:
        nn_layer_activate_lrn(layer, input, output);
        break;

        case NN_CV:
        nn_layer_activate_convolutional(layer, input, output);
        break;

        case NN_FC:
        nn_layer_activate_fully_connected(layer, input, output);
        break;

        case NN_SC:
        nn_layer_activate_singly_connected(layer, input, output);
        break;
    }
}

void nn_layer_destroy(nn_layer_type_t type, void* layer)
{
    switch(type) {
        case NN_LRN:
        nn_layer_destroy_lrn(layer);
        break;

        case NN_CV:
        nn_layer_destroy_convolutional(layer);
        break;

        case NN_FC:
        nn_layer_destroy_fully_connected(layer);
        break;

        case NN_SC:
        nn_layer_destroy_singly_connected(layer);
        break;
    }
}

int nn_layer_input_count(nn_layer_type_t type, void* layer)
{
    switch(type) {
        case NN_LRN:
        return ((nn_layer_lrn_t*)layer)->inputCount;
        break;

        case NN_CV:
        return ((nn_layer_convolutional_t*)layer)->inputCount;
        break;

        case NN_FC:
        return ((nn_layer_fully_connected_t*)layer)->inputCount;
        break;

        case NN_SC:
        return ((nn_layer_singly_connected_t*)layer)->inputCount;
        break;
    }

    return 0;
}

int nn_layer_output_count(nn_layer_type_t type, void* layer)
{
    switch(type) {
        case NN_LRN:
        return ((nn_layer_lrn_t*)layer)->outputCount;
        break;

        case NN_CV:
        return ((nn_layer_convolutional_t*)layer)->outputCount;
        break;

        case NN_FC:
        return ((nn_layer_fully_connected_t*)layer)->outputCount;
        break;

        case NN_SC:
        return ((nn_layer_singly_connected_t*)layer)->outputCount;
        break;
    }

    return 0;
}

int nn_layer_input_dimension_count(nn_layer_type_t type, void* layer)
{
    switch(type) {
        case NN_LRN:
        return ((nn_layer_lrn_t*)layer)->inputDimensionCount;
        break;

        case NN_CV:
        return ((nn_layer_convolutional_t*)layer)->inputDimensionCount;
        break;

        case NN_FC:
        return 1;
        break;

        case NN_SC:
        return 1;
        break;
    }
}

int nn_layer_output_dimension_count(nn_layer_type_t type, void* layer)
{
    switch(type) {
        case NN_LRN:
        return ((nn_layer_lrn_t*)layer)->outputDimensionCount;
        break;

        case NN_CV:
        return ((nn_layer_convolutional_t*)layer)->outputDimensionCount;
        break;

        case NN_FC:
        return 0;
        break;

        case NN_SC:
        return 0;
        break;
    }
}

int* nn_layer_input_dimensions(nn_layer_type_t type, void* layer)
{
    switch(type) {
        case NN_LRN:
        return ((nn_layer_lrn_t*)layer)->inputDimensions;
        break;

        case NN_CV:
        return ((nn_layer_convolutional_t*)layer)->inputDimensions;
        break;

        case NN_FC:
        return NULL;//&((nn_layer_fully_connected_t*)layer)->inputCount;
        break;

        case NN_SC:
        return NULL; //&((nn_layer_singly_connected_t*)layer)->inputCount;
        break;
    }
}

int* nn_layer_output_dimensions(nn_layer_type_t type, void* layer)
{
    switch(type) {
        case NN_LRN:
        return ((nn_layer_lrn_t*)layer)->outputDimensions;
        break;

        case NN_CV:
        return ((nn_layer_convolutional_t*)layer)->outputDimensions;
        break;

        case NN_FC:
        return NULL; //&((nn_layer_fully_connected_t*)layer)->outputCount;
        break;

        case NN_SC:
        return NULL; //&((nn_layer_singly_connected_t*)layer)->outputCount;
        break;
    }
}
