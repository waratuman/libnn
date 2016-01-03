#include "minunit.h"
#include "utils.h"
#include "layer.h"
#include "layers/convolutional.h"

char* test_nn_ii2di()
{
    // Test a 5 x 5 x 2 space
    for (int i = 0; i < 5 * 5 * 2; i++) {
        int result[3];
        int dimensions[3] = { 5, 5, 2 };
        int expected[3] = { 0, 0, 0 };

        nn_ii2di(3, dimensions, i, result);
        expected[0] = i / (5 * 2);
        expected[1] = (i - (expected[0] * (5 * 2))) / 2;
        expected[2] = (i - (expected[0] * (5 * 2)) - (expected[1] * 2));
        // printf("%i, (%i, %i, %i)\t(%i, %i, %i)\n", i, expected[0], expected[1], expected[2], result[0], result[1], result[2]);

        for (int j = 0; j < 3; j++) {
            mu_assert(expected[j] == result[j], "Incorrect mapping for nn_ii2di");
        }
    }
    return NULL;
}

char* test_nn_di2ii()
{
    int dimensions[3] = { 5, 5, 2 };
    

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 2; k++) {
                int input[3] = { i, j, k };
                int result = i * 2 * 5 + j * 2 + k;
                mu_assert(result == nn_di2ii(3, dimensions, input), "Incorrect mapping for nn_di2ii");
            }
        }
    }

    return NULL;
}

char* test_nn_layer_init_convolutional()
{
    nn_layer_convolutional_t* l = calloc(1, sizeof(nn_layer_convolutional_t));
    nn_layer_init_convolutional(l);
    mu_assert(l->outputCount == 0, "outputCount initalize to 0");

    int dim[2] = {5, 5};
    int pad[2] = {0, 0};
    int size[2] = {3, 3};
    int stride[2] = {1, 1};
    int odim[3] = {2, 0, 0};
    l->kernelCount = 2;
    l->inputCount = 25;
    l->inputDimensionCount = 2;
    l->inputDimensions = dim;
    l->outputDimensions = odim;
    l->padding = pad;
    l->size = size;
    l->stride = stride;
    nn_layer_init_convolutional(l);

    mu_assert(l->outputCount == 9 * l->kernelCount, "outputCount initalize to 9");
    mu_assert(l->outputDimensions[0] == l->kernelCount, "outputCount initalize to 9");
    mu_assert(l->outputDimensions[1] == 3, "outputCount initalize to 9");
    mu_assert(l->outputDimensions[2] == 3, "outputCount initalize to 9"); // # of kernels

    return NULL;
}

char* test_nn_layer_create_convolutional()
{
    int dim[3] = {5, 5, 2};
    int pad[3] = {1, 1, 0};
    int siz[3] = {3, 3, 1};
    int str[3] = {1, 1, 1};

    nn_layer_convolutional_t* l = nn_layer_create_convolutional(
        linear_activation, sum_of_products_integration, 50, 3, 1, dim, pad, str, siz);

    mu_assert(l->inputCount == 50, "inputCount init to 25");
    mu_assert(l->outputCount == 50, "outputCount init to 50");
    mu_assert(l->inputDimensionCount == 3, "dimensionCount init to 3");
    mu_assert(l->outputDimensions[0] == l->kernelCount, "outputDimensions[0] == kernelCount")
    for (int i = 0; i < 3; i++) {
        mu_assert(l->inputDimensions[i] == dim[i], "inputDimensions to be intialized");
        mu_assert(l->outputDimensions[i + 1] == (dim[i] + 2 * pad[i] - siz[i]) / str[i] + 1, "inputDimensions to be intialized");
        mu_assert(l->padding[i] == pad[i], "Padding to be intialized");
        mu_assert(l->size[i] == siz[i], "Size to be intialized");
        mu_assert(l->stride[i] == str[i], "Stride to be intialized");
    }

    mu_assert(l->weightCount == 9, "weightCount to be initialized");
    for (int i = 0; i < l->kernelCount; i++) {
        for (int j = 0; j < l->weightCount; j++) {
            mu_assert(l->weights[i][j] == 0.0, "weights to be initalized");
        }
    }

    mu_assert(l->biases[0] == 0.0, "bias to be initalized");
    return NULL;
}

char* test_nn_layer_activate_convolutional()
{
    int d1[2] = {3,3};
    int p1[2] = {0,0};
    int t1[2] = {1,1};
    int s1[2] = {2,2};
    nn_layer_convolutional_t* l1 = nn_layer_create_convolutional(linear_activation, sum_of_products_integration, 9, 2, 2, d1, p1, t1, s1);

    float biases[2] = {1, 0};
    l1->biases = biases;
    for (int k = 0; k < l1->kernelCount; k++) {
        for (int i = 0; i < 4; i++) {
            l1->weights[k][i] = 1;
        }
    }

    float input[9] = { 0, 1, 0,
                       1, 0, 1,
                       0, 1, 0 };

    float output[l1->outputCount];
    nn_layer_activate_convolutional(l1, input, output);

    for (int i = 0; i < 4; i++) {
        mu_assert(output[i] == 3.0, "Convolutional output should = {3, 3, 3, 3}");
    }
    for (int i = 4; i < 8; i++) {
        mu_assert(output[i] == 2.0, "Convolutional output should = {3, 3, 3, 3}");
    }

    return NULL;
}

char* test_nn_layer_activate_padded_convolutional()
{
    int d1[2] = {3,3};
    int p1[2] = {1,1};
    int t1[2] = {1,1};
    int s1[2] = {2,2};
    nn_layer_convolutional_t* l1 = nn_layer_create_convolutional(linear_activation, sum_of_products_integration, 9, 2, 1, d1, p1, t1, s1);

    float biases[1] = {0};
    l1->biases = biases;
    for (int k = 0; k < l1->kernelCount; k++) {
        for (int i = 0; i < 4; i++) {
            l1->weights[k][i] = 1;
        }
    }

    float input[9] = {
        0, 1, 0,
        1, 0, 1,
        0, 1, 0
    };

    float expected_output[16] = {
        0, 1, 1, 0,
        1, 2, 2, 1,
        1, 2, 2, 1,
        0, 1, 1, 0
    };

    float output[l1->outputCount];
    nn_layer_activate_convolutional(l1, input, output);

    for (int i = 0; i < 16; i++) {
        mu_assert(output[i] == expected_output[i], "Convolutional output should = {0, 1, 1...}");
    }

    float biases2[1] = {1};
    l1->biases = biases2;
    nn_layer_activate_convolutional(l1, input, output);
    for (int i = 0; i < 16; i++) {
        mu_assert(output[i] == expected_output[i] + 1, "Convolutional output should = {1, 2, 2...}");
    }
    

    return NULL;
}

char* test_nn_layer_activate_maxpool_convolutional()
{
    int d1[2] = {4,4};
    int p1[2] = {0,0};
    int t1[2] = {2,2};
    int s1[2] = {2,2};
    nn_layer_convolutional_t* l1 = nn_layer_create_convolutional(linear_activation, sum_of_products_integration, 9, 2, 1, d1, p1, t1, s1);
    l1->integration = max_integration;
    float input[16] = { 0, 1, 2, 0,
                       1, 0, 0, 3,
                       0, 0, 1,-3,
                      -1, 0, 4, 0 };
    float expected_output[4] = { 1, 3, 0, 4 };
    float output[l1->outputCount];
    nn_layer_activate_convolutional(l1, input, output);

    for (int i = 0; i < 4; i++) {
        mu_assert(output[i] == expected_output[i], "MaxPool output should = {1, 3, 0, 4}");
    }

    return NULL;
}

char* test_nn_layer_convolutional_is_index_padding()
{
    int d1[2] = {2,2};
    int p1[2] = {0,0};
    int t1[2] = {1,1};
    int s1[2] = {2,2};
    nn_layer_convolutional_t* l1 = nn_layer_create_convolutional(linear_activation, sum_of_products_integration, 4, 2, 1, d1, p1, t1, s1);

    int in[2] = { -1, -1 };
    mu_assert(nn_layer_convolutional_is_index_padding(l1, in) == true, "Expected index to be included in padding");

    in[0] = -1;
    in[1] = 0;
    mu_assert(nn_layer_convolutional_is_index_padding(l1, in) == true, "Expected index to be included in padding");

    in[0] = 0;
    in[1] = -1;
    mu_assert(nn_layer_convolutional_is_index_padding(l1, in) == true, "Expected index to be included in padding");

    in[0] = 0;
    in[1] = 0;
    mu_assert(nn_layer_convolutional_is_index_padding(l1, in) == false, "Expected index to be included in padding");

    in[0] = 2;
    in[1] = 2;
    mu_assert(nn_layer_convolutional_is_index_padding(l1, in) == true, "Expected index to be included in padding");

    in[0] = 2;
    in[1] = 1;
    mu_assert(nn_layer_convolutional_is_index_padding(l1, in) == true, "Expected index to be included in padding");

    in[0] = 1;
    in[1] = 2;
    mu_assert(nn_layer_convolutional_is_index_padding(l1, in) == true, "Expected index to be included in padding");

    in[0] = 1;
    in[1] = 1;
    mu_assert(nn_layer_convolutional_is_index_padding(l1, in) == false, "Expected index to be included in padding");

    return NULL;
}

char* test_nn_layer_create_connected()
{
    nn_layer_convolutional_t* l = nn_layer_create_connected(linear_activation, sum_of_products_integration, 1, 1);
    mu_assert(l->inputCount == 1, "inputCount initalize to 1");
    mu_assert(l->outputCount == 1, "outputCount initalize to 1");
    mu_assert(l->activation == linear_activation, "linear_activation activation function");
    mu_assert(l->biases[0] == 0.0, "1 bias");
    mu_assert(l->weights[0][0] == 0.0, "1 weight");
    
    float* output = calloc(1, sizeof(float));
    float input[1] = {1};
    nn_layer_activate_convolutional(l, input, output);
    mu_assert(output[0] == 0.0, "Expected output to be 0.0");
    free(output);
    
    return NULL;
}

char *all_tests() {
    mu_suite_start();

    mu_run_test(test_nn_ii2di);
    mu_run_test(test_nn_di2ii);
    mu_run_test(test_nn_layer_init_convolutional);
    mu_run_test(test_nn_layer_create_convolutional);
    mu_run_test(test_nn_layer_activate_convolutional);
    mu_run_test(test_nn_layer_activate_padded_convolutional);
    mu_run_test(test_nn_layer_convolutional_is_index_padding);
    mu_run_test(test_nn_layer_activate_maxpool_convolutional);
    mu_run_test(test_nn_layer_create_connected);

    return NULL;
}

RUN_TESTS(all_tests);



// // Convolutional layers apply a kernel to the input and produce a resulting output.
// typedef struct {
//     int inputs;          // Number of inputs
//     int outputs;         // Number of outputs
//
//     nn_kernel_t* kernel; // The kernel to apply
// } nn_layer_convolutional_t;
//
// void nn_layer_init_convolutional(nn_layer_convolutional_t *layer);
// nn_layer_convolutional_t* nn_layer_create_convolutional(nn_kernel_t* kernel, int inputs);
// void nn_layer_destroy_convolutional(nn_layer_convolutional_t* layer);
// void nn_layer_activate_convolutional(nn_layer_convolutional_t *l, float* input, float* output);
