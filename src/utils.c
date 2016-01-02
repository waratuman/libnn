#include "utils.h"
#include <cblas.h>
#include <stdlib.h>

float nn_sdot(int n, float* a, int as, float* b, int bs) {
    return cblas_sdot(n, a, as, b, bs);
}

void nn_shad(int n, float alpha, float* a, int as, float* b, int bs, float* c, int cs) {
    for (int i = 0; i < n; i++) {
        c[i * cs] = alpha * a[i * as] * b[i * bs];
    }
}

// void nn_saxpy()

// Convert an 1D input index to an N dimensional index
// The returned array has the same number of elements as the dimensions array
// does. The caller assumes responsibilty of freeing the returned array.
void nn_ii2di(int dimensionCount, int* dimensions, int ii, int* di)
{
    // int* di = calloc(dimensionCount, sizeof(int));
    int m = 1;

    for (int i = 0; i < dimensionCount; i++) {
        m = m * dimensions[i];
    }

    for (int i = 0; i < dimensionCount; i++) {
        m = m / (dimensions[i]);
        di[i] = ii / m;
        ii = ii - m * di[i];
    }

    // return di;
}

// Convert an N dimensional index to a 1D input index
// The reverse function of nn_ii2di;
int nn_di2ii(int dimensionCount, int* dimensions, int* di)
{
    int ii = 0, m = 1;

    for (int i = dimensionCount - 1; i >= 0; i--) {
        ii = ii + di[i] * m;
        m = m * dimensions[i];
    }

    return ii;
}
