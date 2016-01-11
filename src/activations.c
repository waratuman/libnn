#include "math.h"
#include "activations.h"

// 0: f(x) = x
// 1: f(x) = 1
float linear_activation(float* args, int d)
{
    if (d == 1) {
        return 1.0;
    }

    return args[0];
}

// 0: f(x) &= x^2
// 1: f'(x) &= 2x
float squared_activation(float* args, int d)
{
    if (d == 1) {
        return 2.0 * args[0];
    }

    return pow(args[0], 2);
}

// 0: f(x) = \frac{1}{1 + e^{-x}}
// 1: f'(x) = \frac{1}{1 + e^{-x}} \left(1 - \frac{1}{1 + e^{-x}} \right)
float sigmoid_activation(float* args, int d)
{
    if (d == 1) {
        float x2 = sigmoid_activation(args, 0);
        return x2 * (1.0 - x2);
    }

    return 1.0 / (1.0 + exp(-args[0]));
}

// 0: f(x) = \left \{	\begin{array}{rcl}
//      0 & \mbox{for} & x < 0\\
//      x & \mbox{for} & x \ge 0\end{array} \right.
// 1: f'(x) = \left \{	\begin{array}{rcl}
//      0 & \mbox{for} & x < 0\\
//      1 & \mbox{for} & x \ge 0\end{array} \right.
float rectified_linear(float* args, int d)
{
    if (d == 1) {
        if (args[0] < 0) {
            return 0;
        } else {
            return 1;
        }
    }

    if (args[0] < 0) {
        return 0;
    } else {
        return args[0];
    }
}

// f(\alpha, x)  &= \begin{cases}
//     x & \mbox{if } x > 0 \\
//     \alpha x & \mbox{otherwise}
// \end{cases}\\
// f'(\alpha, x)  &= \begin{cases}
//     1 & \mbox{if } x > 0 \\
//     \alpha & \mbox{otherwise}
// \end{cases}
float leaky_rectified_linear_unit(float* args, int d)
{
    if (d == 1) {
        if (args[1] > 0) {
            return 1;
        } else {
            return args[0];
        }
    }

    if (args[1] > 0) {
        return args[1];
    } else {
        return args[0] * args[1];
    }
}

