#include "minunit.h"
#include "utils.h"
#include "layer.h"
#include "layers/connected.h"

char* test_nn_layer_init_connected()
{
    nn_layer_connected_t* l = calloc(1, sizeof(nn_layer_connected_t));
    
    mu_assert(l->inputCount == 0, "inputCount initalize to 0");
    mu_assert(l->outputCount == 0, "outputCount initalize to 0");
    mu_assert(l->activation == NULL, "NULL activation function");
    mu_assert(l->biases == NULL, "NULL biases");
    mu_assert(l->weights == NULL, "NULL biases");
    return NULL;
}

char* test_nn_layer_create_connected()
{
    nn_layer_connected_t* l = nn_layer_create_connected(linear_activation, 1, 1);
    mu_assert(l->inputCount == 1, "inputCount initalize to 1");
    mu_assert(l->outputCount == 1, "outputCount initalize to 1");
    mu_assert(l->activation == linear_activation, "linear_activation activation function");
    mu_assert(l->biases[0] == 0.0, "1 bias");
    mu_assert(l->weights[0][0] == 0.0, "1 weight");
    return NULL;
}

char* test_nn_layer_activate_connected()
{
    nn_layer_connected_t* l = nn_layer_create_connected(linear_activation, 1, 1);

    float* output = calloc(1, sizeof(float));
    float input[1] = {1};
    nn_layer_activate_connected(l, input, output);
    mu_assert(output[0] == 0.0, "Expected output to be 0.0");
    free(output);
    return NULL;
}

char *all_tests() {
    mu_suite_start();

    mu_run_test(test_nn_layer_init_connected);
    mu_run_test(test_nn_layer_create_connected);
    mu_run_test(test_nn_layer_activate_connected);

    return NULL;
}

RUN_TESTS(all_tests);