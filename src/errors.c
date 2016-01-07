#include "math.h"

// f(a, b) = 1 / 2 (b - a) ^ 2
// f'(a, b) = (b - a)
float squared_error(float a, float b, int d)
{
    if (d == 1) {
        return (b - a);
    }

    return 0.5 * pow(b - a, 2);
}

float sum_error_integration(int n, float* a)
{
    float r;
    for (int i = 0; i < n; i++) {
        r = r + a[i];
    }
    return r / (float)n;
}