#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

// https://en.wikipedia.org/wiki/Activation_function
// The derivative paramter is the nth derivative of the function:
// - 0 is the function
// - 1 is the first derivative
// - 2 is the 2nd derivative
// - -1 is the first integral
// - -2 is the sencond integral
// etc..
typedef float (*nn_activation_fn)(float x, int derivative);

// Identity / Linear
// f(x) = x
// f'(x) = 1
float linear_activation(float x, int derivative);

// Binary Step
// f(x) = \left \{	\begin{array}{rcl}
//     & \mbox{for} & x < 0\\
//     1 & \mbox{for} & x \ge 0\end{array} \right.
// f'(x) = \left \{    \begin{array}{rcl}
//     0 & \mbox{for} & x \ne 0\\
//     ? & \mbox{for} & x = 0\end{array} \right.
float binary_step(float x, int derivative);

// Logistic / Sigmoid / SoftStep
// f(x)=\frac{1}{1+e^{-x}}
// f'(x)=f(x)(1-f(x))
float sigmoid_activation(float x, int derivative);

// TanH
// f(x)=\tanh(x)=\frac{2}{1+e^{-2x}}-1
// f'(x)=1-f(x)^2
float tanh_activation(float x, int derivative);

// ArcTan
// f(x)=\tan^{-1}(x)
// f'(x)=\frac{1}{x^2+1}

// Rectified Linear
// f(x) = \left \{	\begin{array}{rcl}
//     0 & \mbox{for} & x < 0\\
//     x & \mbox{for} & x \ge 0\end{array} \right.
// f'(x) = \left \{	\begin{array}{rcl}
//     0 & \mbox{for} & x < 0\\
//     1 & \mbox{for} & x \ge 0\end{array} \right.
float rectified_linear(float x, int derivative);

// SoftPlus
// f(x)=\log_e(1+e^x)
// f'(x)=\frac{1}{1+e^{-x}}

// Bent Identity
// f(x)=\frac{\sqrt{x^2 + 1} - 1}{2} + x
// f'(x)=\frac{x}{2\sqrt{x^2 + 1}} + 1

// SoftExponential
// f(\alpha,x) = \left \{	\begin{array}{rcl}
//     -\frac{\log_e(1-\alpha (x + \alpha))}{\alpha} & \mbox{for} & \alpha < 0\\
//     x & \mbox{for} & \alpha = 0\\
//     \frac{e^{\alpha x} - 1}{\alpha} + \alpha & \mbox{for} & \alpha > 0\end{array}\right.
// f'(\alpha,x) = \left \{	\begin{array}{rcl}
//     \frac{1}{1-\alpha (\alpha + x)} & \mbox{for} & \alpha < 0\\
//     e^{\alpha x} & \mbox{for} & \alpha \ge 0\end{array} \right.

// Sinusoid
// f(x)=\sin(x)
// f'(x)=\cos(x)

// Sinc
// f(x)=\left \{	\begin{array}{rcl}
//     1 & \mbox{for} & x = 0\\
//     \frac{\sin(x)}{x} & \mbox{for} & x \ne 0\end{array} \right.
// f'(x)=\left \{	\begin{array}{rcl}
//     0 & \mbox{for} & x = 0\\
//     \frac{\cos(x)}{x} - \frac{\sin(x)}{x^2} & \mbox{for} & x \ne 0\end{array} \right.

// Gaussian
// f(x)=e^{-x^2}
// f'(x)=-2xe^{-x^2}

// Noisy Rectified Lineau Unit
// https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
// Y \sim \mathcal{N}(0, \sigma(x))
// f(x) = \max(0, x + Y)
// f'(x) = 

// Leaky Rectified Lineau Unit
// https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
// f(x)  = \begin{cases}
//     x & \mbox{if } x > 0 \\
//     a x & \mbox{otherwise}
// \end{cases}


// Softmax
// https://en.wikipedia.org/wiki/Softmax_function

#endif