#include "math.h"
#include "activations.h"

// 0: f(x) = x
// 1: f(x) = 1
float linear_activation(float x, int d)
{
    if (d == 1) {
        return 1.0;
    }

    return x;
}

// 0: f(x) = \frac{1}{1 + e^{-x}}
// 1: f'(x) = \frac{1}{1 + e^{-x}} \left(1 - \frac{1}{1 + e^{-x}} \right)
float sigmoid_activation(float x, int d)
{
    if (d == 1) {
        float x2 = sigmoid_activation(x, 0);
        return x2 * (1.0 - x2);
    }

    return 1.0 / (1.0 + exp(-x));
}

// 0: f(x) = \left \{	\begin{array}{rcl}
//      0 & \mbox{for} & x < 0\\
//      x & \mbox{for} & x \ge 0\end{array} \right.
// 1: f'(x) = \left \{	\begin{array}{rcl}
//      0 & \mbox{for} & x < 0\\
//      1 & \mbox{for} & x \ge 0\end{array} \right.
float rectified_linear(float x, int d)
{
    if (d == 1) {
        if (x < 0) {
            return 0;
        } else {
            return 1;
        }
    }

    if (x < 0) {
        return 0;
    } else {
        return x;
    }
}
