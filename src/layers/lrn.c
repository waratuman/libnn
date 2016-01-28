#include <nn.h>
#include <math.h>
#include <stdlib.h>

#include "utils.h"

void nn_layer_init_lrn(nn_layer_lrn_t *l)
{
    // l->activation = ;
    l->aggregation = nn_sos_fn;
    l->outputCount = l->inputCount;
    l->outputDimensionCount = l->inputDimensionCount;

    l->kernelInputCount = 1;
    for (int i = 0; i < l->inputDimensionCount; i++) {
        l->outputDimensions[i] = l->inputDimensions[i];
        l->kernelInputCount *= l->size[i];
    }
}

nn_layer_lrn_t*  nn_layer_create_lrn(int ic, int dc, int* dims, int* ks, float k, float alpha, float beta)
{
    nn_layer_lrn_t* l = calloc(1, sizeof(nn_layer_lrn_t));
    l->inputCount = ic;
    l->inputDimensionCount = dc;
    l->k = k;
    l->alpha = alpha;
    l->beta = beta;

    l->inputDimensions = calloc(dc, sizeof(int));
    l->outputDimensions = calloc(dc, sizeof(int));
    l->size = calloc(dc, sizeof(int));

    for (int i = 0; i < dc; i++) {
        l->inputDimensions[i] = dims[i];
        l->size[i] = ks[i];
    }

    nn_layer_init_lrn(l);
    return l;
}

void nn_layer_destroy_lrn(nn_layer_lrn_t* l)
{
    free(l->size);
    free(l->outputDimensions);
    free(l->inputDimensions);
    free(l);
}

void nn_layer_aggregate_lrn(nn_layer_lrn_t *l, float* input, float* output)
{
    int** kernelTransform = calloc(l->kernelInputCount, sizeof(int*));
    for (int i = 0; i < l->kernelInputCount; i++) {
        kernelTransform[i] = calloc(l->inputDimensionCount, sizeof(int));
    }
    nn_kernel_offset(l->inputDimensionCount, l->size, kernelTransform);

    int* kernelCenter = calloc(l->inputDimensionCount, sizeof(int));
    nn_kernel_center(l->inputDimensionCount, l->size, kernelCenter);

    float* kernelInput = calloc(l->kernelInputCount, sizeof(float));
    for (int i = 0; i < l->outputCount; i++) {
        int outputIndex[l->outputDimensionCount];
        nn_ii2di(l->outputDimensionCount, l->outputDimensions, i, outputIndex);

        int* inputCenter = outputIndex;

        // inputCenter times every element in the kernel transform gives us our inputIndexes
        for (int j = 0; j < l->kernelInputCount; j++) {
            int inputIndex[l->inputDimensionCount];
            for (int k = 0; k < l->inputDimensionCount; k++) {
                inputIndex[k] = inputCenter[k] + kernelTransform[j][k];
            }

            if (nn_layer_is_lrn_index_padding(l, inputIndex)) {
                kernelInput[j] = 0.0;
            } else { // else use the input
                kernelInput[j] = input[nn_di2ii(l->inputDimensionCount, l->inputDimensions, inputIndex)];
            }
        }

        output[i] = l->aggregation(l->kernelInputCount, &kernelInput);
    }
    free(kernelInput);

    free(kernelCenter);
    for (int i = 0; i < l->kernelInputCount; i++) {
        free(kernelTransform[i]);
    }
    free(kernelTransform);
}

// b_{x,y}^i = \frac{a_{x,y}^i}
// {
//   \left(
//     k + \alpha \sum_{j=\max(0, i-n/2)}^{\min(N-1,i+n/2)}\left(a_{x,y}^j\right)^2
//   \right)^\beta
// }
void nn_layer_activate_lrn(nn_layer_lrn_t *l, float* input,  float* output)
{
    nn_layer_aggregate_lrn(l, input, output);
    for (int i = 0; i < l->outputCount; i++) {
        output[i] = input[i] / pow(l->k + l->alpha * output[i], l->beta);
    }
}

bool nn_layer_is_lrn_index_padding(nn_layer_lrn_t *l, int* in)
{
    for (int i = 0; i < l->inputDimensionCount; i++) {
        if (in[i] < 0 || in[i] >= l->inputDimensions[i]) {
            return true;
        }
    }

    return false;
}
