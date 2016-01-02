#include "math.h"
#include "activations.h"

// x
float linear_activation(float x)
{
    return x;
}

// 1
float linear_activation_derivative(__attribute__ ((unused)) float x)
{
    return 1.0;
}

// \frac{1}{1 + e^{-x}}
float sigmoid_activation(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

// \frac{1}{1 + e^{-x}} \left(1 - \frac{1}{1 + e^{-x}} \right)
float sigmoid_activation_derivative(float x)
{
    float x2 = sigmoid_activation(x);
    return x2 * (1.0 - x2);
}

// f(x) = \left \{	\begin{array}{rcl}
//     0 & \mbox{for} & x < 0\\
//     x & \mbox{for} & x \ge 0\end{array} \right.
float rectified_linear(float x)
{
    if (x < 0) {
        return 0;
    } else {
        return x;
    }
}

// f'(x) = \left \{	\begin{array}{rcl}
//     0 & \mbox{for} & x < 0\\
//     1 & \mbox{for} & x \ge 0\end{array} \right.
float rectified_linear_derivative(float x)
{
    if (x < 0) {
        return 0;
    } else {
        return 1;
    }
}
