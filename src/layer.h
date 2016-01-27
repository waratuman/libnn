// Ref: http://page.mi.fu-berlin.de/rojas/neural/
#pragma once

typedef enum {
    LRN,
    SINGLY_CONNECTED,
    FULLY_CONNECTED,
    CONVOLUTIONAL
} LAYER_TYPE;

void nn_layer_init(void* layer);

void nn_layer_destroy(LAYER_TYPE type, void* layer);

void nn_layer_integrate(LAYER_TYPE type, void* layer, float* input, float* output);

void nn_layer_activate(LAYER_TYPE type, void* layer, float* input, float* output);

void nn_layer_derivate(LAYER_TYPE type, void* layer, float* input, float* output);


int nn_layer_input_count(LAYER_TYPE type, void* layer);

int nn_layer_output_count(LAYER_TYPE type, void* layer);

int nn_layer_input_dimension_count(LAYER_TYPE type, void* layer);

int nn_layer_output_dimension_count(LAYER_TYPE type, void* layer);

int* nn_layer_input_dimensions(LAYER_TYPE type, void* layer);

int* nn_layer_output_dimensions(LAYER_TYPE type, void* layer);
