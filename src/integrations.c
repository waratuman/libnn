#include "utils.h"
#include "math.h"

float sum_of_products_integration(int n, float** args)
{
    return nn_sdot(n, args[0], 1, args[1], 1);
}

float euclidean_integration(int n, float** args)
{
    float result = 0;
    for (int i = 0; i < n; i++) {
        result = result + pow(args[0][i] - args[1][i], 2);
    }
    return result * (1 / (2 * (float)n));
}

float sum_of_squares(int n, float** args)
{
    float result = 0;
    for (int i = 0; i < n; i++) {
        result = result + pow(args[0][i], 2);
    }
    return result;
}

float max_integration(int n, float** args)
{
    int max = 0;
    for (int i = 0; i < n; i++) {
        if (args[0][i] > args[0][max]) {
            max = i;
        }
    }
    return args[0][max];
}

float avg_integration(int n, float** args)
{
    float avg = 0.0;
    for (int i = 0; i < n; i++) {
        avg += args[0][i];
    }
    return avg / (float)n;
}