#include <math.h>

#include "utils.h"

// Sum of Products Aggregation Function
float nn_sop_fn(int n, float** args)
{
    return nn_sdot(n, args[0], 1, args[1], 1);
}

// Euclidean Aggregation Function
float nn_euclidean_fn(int n, float** args)
{
    float result = 0;
    for (int i = 0; i < n; i++) {
        result = result + pow(args[0][i] - args[1][i], 2);
    }
    return result * (1 / (2 * (float)n));
}

// Sum of Squares Aggregation Function
float nn_sos_fn(int n, float** args)
{
    float result = 0;
    for (int i = 0; i < n; i++) {
        result = result + pow(args[0][i], 2);
    }
    return result;
}

// Max Aggregation Function
float nn_max_fn(int n, float** args)
{
    int max = 0;
    for (int i = 0; i < n; i++) {
        if (args[0][i] > args[0][max]) {
            max = i;
        }
    }
    return args[0][max];
}

// Average Aggregation Function
float nn_avg_fn(int n, float** args)
{
    float avg = 0.0;
    for (int i = 0; i < n; i++) {
        avg += args[0][i];
    }
    return avg / (float)n;
}
