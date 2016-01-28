#include <math.h>
#include "utils.h"

void nn_kernel_offset(int dimensionCount, int* dimensions, int** result)
{
    int resultCount = 1;
    for (int i = 0; i < dimensionCount; i++) {
        resultCount *= dimensions[i];
    }

    for (int i = 0; i < resultCount; i++) {
        nn_ii2di(dimensionCount, dimensions, i, result[i]);
        for (int j = 0; j < dimensionCount; j++) {
            result[i][j] = result[i][j] - floor(dimensions[j] / 2.0);
        }
    }

}

void nn_kernel_center(int dimesionCount, int* dimesions, int* result)
{
    for (int i = 0; i < dimesionCount; i++) {
        result[i] = dimesions[i] / 2;
    }
}
