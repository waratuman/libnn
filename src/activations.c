#include <nn.h>
#include <math.h>


// Identity / Linear Function
static inline float nn_identity_inline(float* args, int d)
{
    if (d == 1) {
        return 1.0;
    }

    return args[0];
}

float nn_identity_fn(float* args, int d) {
    return nn_identity_inline(args, d);
}

float nn_linear_fn(float* args, int d)
{
    return nn_identity_inline(args, d);
}

// Squared Function
float nn_squared_fn(float* args, int d)
{
    if (d == 1) {
        return 2.0 * args[0];
    }

    return pow(args[0], 2);
}

// Binary Step Function
float nn_binary_step_fn(float* args, int d)
{
    if (d == 1) {
        return 0.0;
    }

    if (args[0] < 0) {
        return 0;
    }

    return 1;
}

// Logistic / Sigmoid / SoftStep Function
static inline float nn_sigmoid_inline(float* args, int d)
{
    if (d == 1) {
        float x2 = nn_sigmoid_inline(args, 0);
        return x2 * (1.0 - x2);
    }

    return 1.0 / (1.0 + exp(-args[0]));
}

float nn_sigmoid_fn(float* args, int d)
{
    return nn_sigmoid_inline(args, d);
}

float nn_logistic_fn(float* args, int d)
{
    return nn_sigmoid_inline(args, d);
}

float nn_softstep_fn(float* args, int d)
{
    return nn_sigmoid_inline(args, d);
}

// TanH Function
float nn_tanh_fn(float* args, int d)
{
    if (d == 1) {
        return 1.0 - pow(nn_tanh_fn(args, 0), 2);
    }

    return tanh(args[0]);
}

// ArcTan Function
float nn_arctan_fn(float *args, int d)
{
    if (d == 1) {
        return 1 / (pow(args[0], 2) + 1);
    }

    return atan(args[0]);
}

// Rectified Linear Unit Function
float nn_relu_fn(float* args, int d)
{
    if (d == 1) {
        return (args[0] < 0) ? 0 : 1;
    }

    return (args[0] < 0) ? 0 : args[0];
}

// Parameteric Rectified Linear Unit / Leaky Rectified Linear Unit Function
float nn_prelu_fn(float* args, int d)
{
    if (d == 1) {
        return (args[1] < 0) ? args[0] : 1;
    }

    return (args[1] < 0) ? args[1] * args[0] : args[1];
}

// Exponential Linear Unit Function
float nn_elu_fn(float* args, int d)
{
    if (d == 1) {
        return (args[1] < 0) ? nn_elu_fn(args, 0) + args[0] : 1.0;
    }

    return (args[1] < 0) ? args[0] * (exp(args[1]) - 1) : args[1];
}

// SoftPlus Function
float nn_softplus_fn(float* args, int d)
{
    if (d == 1) {
        return 1.0 / (1.0 + exp(-args[0]));
    }

    return log(1 + exp(args[0]));
}

// Bent Identity Function
float nn_bent_identity_fn(float* args, int d)
{
    if (d == 1) {
        return args[0] / (2 * sqrt(pow(args[0], 2) + 1)) + 1.0;
    }

    return (sqrt(pow(args[0], 2) + 1) - 1) / 2.0 + args[0];
}

// Soft Exponential Function
float nn_softexp_fn(float* args, int d)
{
    if (d == 1) {
        if (args[0] < 0) {
            return 1 / (1 - args[0] * (args[0] + args[1]));
        }
        return exp(args[0] * args[1]);
    }

    if (args[0] < 0) {
        return -(log(1 - args[0] * (args[1] + args[0])) / args[0]);
    }

    if (args[0] == 0.0) {
        return args[1];
    }

    return (exp(args[0] * args[1]) - 1) / args[0] + args[0];
}

// Sinusoid Function
float nn_sin_fn(float* args, int d)
{
    if (d == 1) {
        return cos(args[0]);
    }

    return sin(args[0]);
}

// Sinc Function
float nn_sinc_fn(float* args, int d)
{
    if (d == 1) {
        if (args[0] == 0.0) {
            return 0;
        }
        return cos(args[0]) / args[0] - sin(args[0]) / pow(args[0], 2);
    }

    if (args[0] == 0.0) {
        return 1;
    }

    return sin(args[0]) / args[0];
}

// Gaussian Function
float nn_gaussian_fn(float* args, int d)
{
    if (d == 1) {
        return -1 * args[0] * exp(pow(-args[0], 2));
    }

    return exp(pow(-args[0], 2));
}

// TODO: Noisy Rectified Linear Unit Function
// https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
// Y &\sim \mathcal{N}(0, \sigma(x))
// f(x) &= \max(0, x + Y)

// TODO: Softmax
// https://en.wikipedia.org/wiki/Softmax_function
