#include "utils.h"
#include "math.h"

float sum_of_products_integration(int n, float* a, float* b)
{
    return nn_sdot(n, a, 1, b, 1);
}

float sum_of_squares_integration(int n, float* a, float* b)
{
    float result = 0;
    for (int i = 0; i < n; i++) {
        result = result + pow(a[i] - b[i], 2);
    }
    return result * (1 / (2 * (float)n));
}

float max_integration(int n, float* a, float* b)
{
    int max = 0;
    for (int i = 0; i < n; i++) {
        if (b[i] > b[max]) {
            max = i;
        }
    }
    return b[max];
}