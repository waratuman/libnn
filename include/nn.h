#pragma once

#include <stdbool.h>

// = Activation Functions ======================================================

/* All activation functions implement the following interface. The function takes
 * two arguments:
 * - args: An array of arguments, most function only expect 1 elemnt
 * - derivative: Calculate the nth derivative / integral of the function
 *   -2: \iint{f(x)dx}
 *   -1: \int{f(x)dx}
 *    0: f(x)
 *    1: f^\prime(x) = \frac{dy}{dx} = \frac{d\left(f(x)\right)}{dx}
 *    2: f^{\prime\prime}(x) = \frac{d^2y}{dx^2} = \frac{d^2\left(f(x)\right)}{dx^2}
 */
typedef float (*nn_activation_fn)(float* args, int derivative);

/* Identity / Linear Function
 * https://en.wikipedia.org/wiki/Activation_function
 * f(x) &= x
 * f^\prime(x) &= 1
 */
float nn_identity_fn(float* args, int derivative);
float nn_linear_fn(float* args, int derivative);

/* Squared Function
 * https://en.wikipedia.org/wiki/Activation_function
 * f(x) &= x^2
 * f^\prime(x) &= 2x
 */
float nn_squared_fn(float* args, int derivative);

/* Binary Step Function
 * https://en.wikipedia.org/wiki/Activation_function
 * f(x) &= \left \{    \begin{array}{rcl}
 *    0 & \mbox{for} & x < 0\\
 *    1 & \mbox{for} & x \ge 0\end{array} \right.
 * f^\prime(x) &= \left \{    \begin{array}{rcl}
 *    0 & \mbox{for} & x \ne 0\\
 *    ? & \mbox{for} & x = 0\end{array} \right.
 */
float nn_binary_step_fn(float* args, int derivative);

/* Logistic / Sigmoid / SoftStep Function
 * https://en.wikipedia.org/wiki/Activation_function
 * f(x) &= \frac{1}{1+e^{-x}}
 * f^\prime(x) &= f(x)(1-f(x))
 */
float nn_sigmoid_fn(float* args, int derivative);
float nn_logistic_fn(float* args, int derivative);
float nn_softstep_fn(float* args, int derivative);

/* TanH Function
 * https://en.wikipedia.org/wiki/Activation_function
 * f(x) &= \tanh(x) &= \frac{2}{1+e^{-2x}} - 1
 * f^\prime(x) &= 1 - f(x)^2
 */
float nn_tanh_fn(float* args, int derivative);

/* ArcTan Function
 * https://en.wikipedia.org/wiki/Activation_function
 * f(x) &= \tan^{-1}(x)
 * f^\prime(x) &= \frac{1}{x^2+1}
 */
float nn_arctan_fn(float *args, int derivative);

/* Rectified Linear Unit Function
 * https://en.wikipedia.org/wiki/Activation_function
 * f(x) &= \left \{ \begin{array}{rcl}
 *     0 & \mbox{for} & x < 0 \\
 *     x & \mbox{for} & x \ge 0\end{array} \right.
 * f^\prime(x) &= \left \{ \begin{array}{rcl}
 *     0 & \mbox{for} & x < 0 \\
 *     1 & \mbox{for} & x \ge 0\end{array} \right.
 */
float nn_relu_fn(float* args, int derivative);

/* Parameteric Rectified Linear Unit / Leaky Rectified Linear Unit Function
 * https://en.wikipedia.org/wiki/Activation_function
 * f(\alpha, x)  &= \begin{cases}
 *    \alpha x & \mbox{if } x < 0 \\
 *     x & \mbox{if } x \ge 0 \\
 * \end{cases}
 * f^\prime(\alpha, x)  &= \begin{cases}
 *    \alpha & \mbox{if } x < 0 \\
 *    1 & \mbox{if } x \ge 0 \\
 * \end{cases}
 */
float nn_prelu_fn(float* args, int derivative);

/* Exponential Linear Unit Function
 * https://en.wikipedia.org/wiki/Activation_function
 * f(\alpha, x)  &= \begin{cases}
 *     \alpha \left( e^x - 1 \right) & \mbox{if } x < 0 \\
 *     x & \mbox{if } x \ge 0 \\
 * \end{cases}
 * f^\prime(\alpha, x)  &= \begin{cases}
 *     f(\alpha, x) + \alpha & \mbox{if } x < 0 \\
 *     1 & \mbox{if } x \ge 0 \\
 * \end{cases}
 */
float nn_elu_fn(float* args, int derivative);

/* SoftPlus Function
 * https://en.wikipedia.org/wiki/Activation_function
 * f(x) &= \log_e(1+e^x)
 * f^\prime(x) &= \frac{1}{1+e^{-x}}
 */
float nn_softplus_fn(float* args, int derivative);

/* Bent Identity Function
 * https://en.wikipedia.org/wiki/Activation_function
 * f(x) &= \frac{\sqrt{x^2 + 1} - 1}{2} + x
 * f^\prime(x) &= \frac{x}{2\sqrt{x^2 + 1}} + 1
 */
float nn_bent_identity_fn(float* args, int derivative);

/* Soft Exponential Function
 * https://en.wikipedia.org/wiki/Activation_function
 * f(\alpha,x) &= \left \{ \begin{array}{rcl}
 *    -\frac{\log_e(1-\alpha (x + \alpha))}{\alpha} & \mbox{for} & \alpha < 0\\
 *    x & \mbox{for} & \alpha = 0\\
 *    \frac{e^{\alpha x} - 1}{\alpha} + \alpha & \mbox{for} & \alpha > 0\end{array}\right.
 *f^\prime(\alpha,x) &= \left \{ \begin{array}{rcl}
 *    \frac{1}{1-\alpha (\alpha + x)} & \mbox{for} & \alpha < 0\\
 *    e^{\alpha x} & \mbox{for} & \alpha \ge 0\end{array} \right.
 */
float nn_softexp_fn(float* args, int derivative);

/* Sinusoid Function
 * https://en.wikipedia.org/wiki/Activation_function
 * f(x) &= \sin(x)
 * f^\prime(x) &= \cos(x)
 */
float nn_sin_fn(float* args, int derivative);

/* Sinc Function
 * https://en.wikipedia.org/wiki/Activation_function
 *f(x) &= \left \{ \begin{array}{rcl}
 *    1 & \mbox{for} & x = 0\\
 *    \frac{\sin(x)}{x} & \mbox{for} & x \ne 0\end{array} \right.
 *f^\prime(x) &= \left \{ \begin{array}{rcl}
 *    0 & \mbox{for} & x = 0\\
 *    \frac{\cos(x)}{x} - \frac{\sin(x)}{x^2} & \mbox{for} & x \ne 0\end{array} \right.
 */
float nn_sinc_fn(float* args, int derivative);

/* Gaussian Function
 * https://en.wikipedia.org/wiki/Activation_function
 * f(x) &= e^{-x^2}
 * f^\prime(x) &= -2xe^{-x^2}
 */
float nn_gaussian_fn(float* args, int derivative);

// TODO: Noisy Rectified Linear Unit Function
// https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
// Y &\sim \mathcal{N}(0, \sigma(x))
// f(x) &= \max(0, x + Y)

// TODO: Softmax
// https://en.wikipedia.org/wiki/Softmax_function

// = Aggregations ==============================================================

/* All aggregation functions implement the following interface. The function takes
 * two arguments:
 * - n: The number of elements in arg
 * - args: An array of array of values
 */
typedef float (*nn_aggregation_fn)(int n, float** args);

/* Sum of Products Aggregation Function
 * \sum_{i=0}^n \left( a_{0,i} a_{1,i} \right)
 */
float nn_sop_fn(int n, float** args);

/* Euclidean Aggregation Function
 * \frac{1}{2n} \sum_{i=0}^{n} \left( a_{0,i} - a_{1,i} \right)^{2}
 */
float nn_euclidean_fn(int n, float** args);

/* Sum of Squares Aggregation Function
 * \sum_{i=0}^n \left( a_{0,i}^{2} \right)
 */
float nn_sos_fn(int n, float** args);

/* Max Aggregation Function
 * \max\left(a_0\right)
 */
float nn_max_fn(int n, float** args);

/* Average Aggregation Function
 * \avg\left(a_0\right)
 */
float nn_avg_fn(int n, float** args);

// = Error Functions ===========================================================

/* The derivative paramter is the nth derivative of the function:
 * - 0 is the function
 * - 1 is the first derivative
 * - 2 is the 2nd derivative
 * - -1 is the first integral
 * - -2 is the sencond integral
 * etc..
 */
typedef float (*nn_error_fn)(float a, float b, int derivative);
typedef float (*nn_error_aggregation_fn)(int n, float* a);

/* Sqaured Error / Mean Squared Error
 * f(a, b) &= 1 / 2 (b - a) ^ 2
 * f\prime(a, b) &= (b - a)
 */
float nn_mse_fn(float a, float b, int d);
float nn_mse_aggregate_fn(int n, float* a);

// = Layer interface ===========================================================

typedef enum nn_layer_type_e {
    NN_LRN, // Localized Response Normalization
    NN_SC,  // Singly Connected
    NN_FC,  // Fully Connected
    NN_CV,  // Convolutional
} nn_layer_type_t;

void nn_layer_init(nn_layer_type_t type, void* layer);

void nn_layer_destroy(nn_layer_type_t type, void* layer);

void nn_layer_aggregate(nn_layer_type_t type, void* layer, float* input, float* output);

void nn_layer_activate(nn_layer_type_t type, void* layer, float* input, float* output);

int nn_layer_input_count(nn_layer_type_t type, void* layer);

int nn_layer_output_count(nn_layer_type_t type, void* layer);

int nn_layer_input_dimension_count(nn_layer_type_t type, void* layer);

int nn_layer_output_dimension_count(nn_layer_type_t type, void* layer);

int* nn_layer_input_dimensions(nn_layer_type_t type, void* layer);

int* nn_layer_output_dimensions(nn_layer_type_t type, void* layer);

/* = Fully Connected Layer =====================================================
 *
 * Fully connected layers connect every input to every output using a aggregate
 * function. For each output a given activation function is applied. The aggregate
 * funciton is of the form:
 * 
 *     typedef float (*nn_aggregation_fn)(int count, float* a, float* b);
 * 
 * And the activation function is of the form:
 * 
 *     typedef float (*nn_activation_fn)(float x);
 * 
 * For each output i the result is:
 * 
 *     o_i &= a \left( g\left( \hat{w_i}, \hat{\imath} \right) + b_i \right)
 * 
 * Where a is the activation function and g is the aggregate function. In a typical
 * fully connected layer the aggregate function is the sum of products:
 * 
 *     o_i &= a \left( g\left( \hat{w_i}, \hat{\imath} \right) + b_i \right)\\
 *     g\left( \hat{w_i}, \hat{\imath} \right) &= \sum_{j=0}^{n} \left( w_{ij}i_j \right)
 * 
 * Where i is the ith output, j is the jth input, n is the number of
 * inputs, w_ij is the weight associated between the ith ouput and jth
 * input, i_j is the jth input and b_i is the bias of the ith output.
 */

typedef struct {
    int inputCount;                     // Number of inputs to the layer
    int outputCount;                    // Number of outputs of the layer
    nn_activation_fn activation;        // Activation function
    nn_aggregation_fn aggregation;      // Aggregate function

    float* biases;
    float** weights;
} nn_layer_fully_connected_t;

void nn_layer_init_fully_connected(nn_layer_fully_connected_t *layer);

nn_layer_fully_connected_t* nn_layer_create_fully_connected(nn_activation_fn activation, nn_aggregation_fn aggregation, int inputCount, int outputCount);

void nn_layer_destroy_fully_connected(nn_layer_fully_connected_t* layer);

void nn_layer_aggregate_fully_connected(nn_layer_fully_connected_t *layer, float* input, float* output);

void nn_layer_activate_fully_connected(nn_layer_fully_connected_t *layer, float* input, float* output);

// = Singly Connected ==========================================================

typedef struct {
    int inputCount;             // Number of inputs
    int outputCount;            // Number of outputs
    int weightCount;            // Number of weights

    nn_activation_fn activation;    // Activation function

    float* biases;           // The bias of the kernel
    float* weights;          // The weights of the kernel
} nn_layer_singly_connected_t;

void nn_layer_init_singly_connected(nn_layer_singly_connected_t *layer);

nn_layer_singly_connected_t* nn_layer_create_singly_connected(nn_activation_fn activation, int inputCount);

void nn_layer_destroy_singly_connected(nn_layer_singly_connected_t* layer);

void nn_layer_aggregate_singly_connected(nn_layer_singly_connected_t *layer, float* input, float* output);

void nn_layer_activate_singly_connected(nn_layer_singly_connected_t *layer, float* input, float* output);

/* = Localized Response Normalization ==========================================
 * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
 * b_{x,y}^i = \frac{a_{x,y}^i}
 * {
 *   \left(
 *     k + \alpha \sum_{j=\max(0, i-n/2)}^{\min(N-1,i+n/2)}\left(a_{x,y}^j\right)^2
 *   \right)^\beta
 * }
 */
typedef struct {
    int inputCount;             // Number of inputs
    int outputCount;            // Number of outputs (== inputCount)
    int kernelInputCount;       // Number of inputs to the kernel
    int inputDimensionCount;    // Number of input dimensions
    int outputDimensionCount;   // Number of output dimensions

    nn_activation_fn activation;    // Activation function
    nn_aggregation_fn aggregation;  // Aggregate function

    float k;                // defaults to 2
    float alpha;            // defaults to 10^-4
    float beta;             // defaults to 0.75

    int* size;              // The size of the kernel in each dimension
    int* inputDimensions;   // The dimensions of the input
    int* outputDimensions;  // The dimensions of the output (== inputDimesions)
} nn_layer_lrn_t;

void nn_layer_init_lrn(nn_layer_lrn_t *layer);

nn_layer_lrn_t* nn_layer_create_lrn(int inputCount, int inputDimensionCount, int* dimensions, int* kernel_size, float k, float alpha, float beta);

void nn_layer_destroy_lrn(nn_layer_lrn_t* layer);

/* The output has an extra dimension (first dimension, index 0),
 * which is the number for kernels.
 */
void nn_layer_aggregate_lrn(nn_layer_lrn_t *l, float* input, float* output);

/* The output has an extra dimension (first dimension, index 0),
 * which is the number for kernels.
 */
void nn_layer_activate_lrn(nn_layer_lrn_t *layer, float* input, float* output);

/* Returns true if the given index (an array of the size layer->inputDimensionCount)
 * lies withing the padded region (outside the input). False if it lies within
 * the input.
 */
bool nn_layer_is_lrn_index_padding(nn_layer_lrn_t *layer, int* index);

// = Convolutional =============================================================

typedef struct {
    int inputCount;             // Number of inputs
    int outputCount;            // Number of outputs
    int weightCount;            // Number of weights
    int kernelCount;            // Number of kernels
    int inputDimensionCount;    // Number of input dimensions
    int outputDimensionCount;   // Number of output dimensions

    nn_activation_fn activation;    // Activation function
    nn_aggregation_fn aggregation;  // Aggregate function

    float* biases;           // The bias of the kernel
    float** weights;         // The weights of the kernel

    int* size;               // The size of the kernel in each dimension
    int* stride;             // The stride in each dimension, defaults to 1
    int* padding;            // The padding in each dimension, defaults to 0
    int* inputDimensions;    // The dimensions of the input
    int* outputDimensions;   // The dimensions of the output
} nn_layer_convolutional_t;

void nn_layer_init_convolutional(nn_layer_convolutional_t *layer);

nn_layer_convolutional_t* nn_layer_create_convolutional(
    nn_activation_fn activation,
    nn_aggregation_fn aggregation,
    int inputCount,
    int inputDimensionCount,
    int kernelCount,
    int* dimensions,
    int* padding,
    int* kernel_stride,
    int* kernel_size
);

void nn_layer_destroy_convolutional(
    nn_layer_convolutional_t* layer
);

/* The output has an extra dimension (first dimension, index 0),
 * which is the number for kernels.
 */
void nn_layer_aggregate_convolutional(
    nn_layer_convolutional_t *l,
    float* input,
    float* output
);

/* The output has an extra dimension (first dimension, index 0),
 * which is the number for kernels.
 */
void nn_layer_activate_convolutional(
    nn_layer_convolutional_t *layer,
    float* input,
    float* output
);

/* TODO: Should not be in the public header file.
 * Returns true if the given index (an array of the size layer->inputDimensionCount)
 * lies withing the padded region (outside the input). False if it lies within
 * the input.
 */
bool nn_layer_is_convolutional_index_padding(
    nn_layer_convolutional_t *layer,
    int* index
);

/* = Network ================================================================ */

typedef struct nn_network_t {
    int layerCount;
    nn_layer_type_t* layerTypes;
    void** layers;

    nn_error_fn error;      // Error / Loss function (defaults to the sum of squares)

    float** activations;    // The stored node activations (for backpropagation)
    float** derivatives;    // The stored node activation derivatives (for backpropagation)
} nn_network_t;

void nn_network_init(nn_network_t *network);

nn_network_t* nn_network_create(
    int layerCount,
    nn_layer_type_t* layerTypes,
    void** layers
);

// Note: Will also call destroy on any layers
void nn_network_destroy(nn_network_t* network);

// Input is the size of the first layer in the network
// Output is of the size of the last layer in the network
void nn_network_activate(nn_network_t *network, float* input, float* output);

float nn_network_loss(nn_network_t* network, float* input, float* target);

void nn_network_train(nn_network_t* network, float* input, float* target);
