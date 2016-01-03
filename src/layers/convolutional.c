#include <stdlib.h>
#include "utils.h"
#include "kernel.h"
#include "convolutional.h"

// \prod_{i=1}^{n}\left(\frac{d_i + 2p_i - k_i}{s_i} + 1\right)
void nn_layer_init_convolutional(nn_layer_convolutional_t *l)
{
    l->outputDimensionCount = l->inputDimensionCount + 1;

    // Initalize the number of ouputs of the layer
    if (l->inputCount == 0) {
        l->outputCount = 0;
    } else {
        l->outputCount = 1;
        
        for (int i = 0; i < l->inputDimensionCount; i++) {
            l->outputDimensions[i] = ((l->inputDimensions[i] + 2 * l->padding[i] - l->size[i]) / l->stride[i] + 1);
            l->outputCount = l->outputCount * l->outputDimensions[i];
        }
        l->outputDimensions[l->outputDimensionCount - 1] = l->kernelCount;
        l->outputCount = l->outputCount * l->kernelCount;
    }

    // TODO: Initalize the biases of the kernels using a better distribution
    l->biases = calloc(l->kernelCount, sizeof(float));

    // TODO: Initalize the weights of the kernel using a better distribution
    l->weightCount = 1;
    for (int i = 0; i < l->inputDimensionCount; i++) {
        l->weightCount *= l->size[i];
    }
    l->weights = calloc(l->kernelCount, sizeof(float*));
    for (int i = 0; i < l->kernelCount; i++) {
        l->weights[i] = calloc(l->weightCount, sizeof(float));
    }

    if (l->integration == NULL) {
        l->integration = sum_of_products_integration;
    }
}

nn_layer_convolutional_t* nn_layer_create_convolutional(nn_activation_fn f, int i, int d, int k, int* ds, int* p, int* st, int* sz)
{
    nn_layer_convolutional_t* l = calloc(1, sizeof(nn_layer_convolutional_t));
    l->activation = f;
    l->inputCount = i;
    l->inputDimensionCount = d;
    l->kernelCount = k;

    l->inputDimensions = calloc(d, sizeof(int));
    l->outputDimensions = calloc(d + 1, sizeof(int));
    l->padding = calloc(d, sizeof(int));
    l->stride = calloc(d, sizeof(int));
    l->size = calloc(d, sizeof(int));

    for (int i = 0; i < d; i++) {
        l->inputDimensions[i] = ds[i];
        l->padding[i] = p[i];
        l->stride[i] = st[i];
        l->size[i] = sz[i];
    }

    nn_layer_init_convolutional(l);
    return l;
}

void nn_layer_destroy_convolutional(nn_layer_convolutional_t* l)
{
    free(l->size);
    free(l->stride);
    free(l->padding);
    free(l->inputDimensions);
    free(l->outputDimensions);
    free(l->biases);

    for (int i = 0; i < l->kernelCount; i++) {
        free(l->weights[i]);
    }
    free(l->weights);

    free(l);
}

void nn_layer_activate_convolutional(nn_layer_convolutional_t *l, float* input, float* output)
{
    int** kernelTransform = calloc(l->weightCount, sizeof(int*));
    for (int i = 0; i < l->weightCount; i++) {
        kernelTransform[i] = calloc(l->inputDimensionCount, sizeof(int));
    }
    nn_kernel_offset(l->inputDimensionCount, l->size, kernelTransform);

    int* kernelCenter = calloc(l->inputDimensionCount, sizeof(int));
    nn_kernel_center(l->inputDimensionCount, l->size, kernelCenter);

    float* kernelInput = calloc(l->weightCount, sizeof(float));
    for (int i = 0; i < l->outputCount; i++) {
        int outputIndex[l->outputDimensionCount];
        nn_ii2di(l->outputDimensionCount, l->outputDimensions, i, outputIndex);


        // inputCenter = a \circ b + c - p, where a = outputIndex, b = stride, c = kernel center, p = padding
        int inputCenter[l->inputDimensionCount];
        for (int j = 0; j < l->inputDimensionCount; j++) {
            inputCenter[j] = outputIndex[j] * l->stride[j] + kernelCenter[j] - l->padding[j];
        }
        
        // inputCenter times every element in the kernel transform gives us our inputIndexes
        for (int j = 0; j < l->weightCount; j++) {
            int inputIndex[l->inputDimensionCount];
            for (int k = 0; k < l->inputDimensionCount; k++) {
                inputIndex[k] = inputCenter[k] + kernelTransform[j][k];
            }

            // If the inputIndex is in the padded region use the pad value (this prevents us from having to move some data around)
            // TODO: Support other kind of padded values, eg extend, wrap, crop
            // right now we only 0 the edges, which might not be the best way to go for all networks
            if (nn_layer_convolutional_is_index_padding(l, inputIndex)) {
                kernelInput[j] = 0.0;
            } else { // else use the input
                kernelInput[j] = input[nn_di2ii(l->inputDimensionCount, l->inputDimensions, inputIndex)];
            }
        }

        int kernelIndex = outputIndex[l->outputDimensionCount - 1];
        float bias = l->biases[kernelIndex];
        float* weights = l->weights[kernelIndex];
        output[i] = l->activation(l->integration(l->weightCount, weights, kernelInput) + bias);
    }
    free(kernelInput);

    for (int i = 0; i < l->weightCount; i++) {
        free(kernelTransform[i]);
    }
    free(kernelTransform);
    free(kernelCenter);
}

bool nn_layer_convolutional_is_index_padding(nn_layer_convolutional_t *l, int* in)
{
    for (int i = 0; i < l->inputDimensionCount; i++) {
        if (in[i] < 0 || in[i] >= l->inputDimensions[i]) {
            return true;
        }
    }

    return false;
}

