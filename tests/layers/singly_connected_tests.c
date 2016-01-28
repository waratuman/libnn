#include <nn.h>

#include "minunit.h"

char* test_nn_layer_create_singly_connected()
{
    nn_layer_singly_connected_t* l = nn_layer_create_singly_connected(nn_linear_fn, 2);
    mu_assert(l->inputCount == 2, "inputCount initalize to 2");
    mu_assert(l->outputCount == 2, "outputCount initalize to 2");
    mu_assert(l->weightCount == 2, "outputCount initalize to 2");
    mu_assert(l->activation == nn_linear_fn, "nn_linear_fn activation function");
    mu_assert(l->biases[0] == 0.0, "2 bias");
    mu_assert(l->biases[1] == 0.0, "2 bias");
    mu_assert(l->weights[0] == 0.0, "2 weights");
    mu_assert(l->weights[1] == 0.0, "2 weights");
    
    float output[2];
    float input[2] = {1, 0};
    nn_layer_activate_singly_connected(l, input, output);
    mu_assert(output[0] == 1.0, "Expected output to be 1.0");
    mu_assert(output[1] == 0.0, "Expected output to be 0.0");

    nn_layer_destroy_singly_connected(l);
    return NULL;
}

char* test_nn_layer_aggregate_singly_connected()
{
    nn_layer_singly_connected_t* l;
    
    l = nn_layer_create_singly_connected(nn_linear_fn, 2);
    l->weights[0] = 1.0;
    l->weights[1] = 1.0;

    float output[2];
    float input[2] = {1, 1};
    nn_layer_aggregate_singly_connected(l, input, output);
    mu_assert(output[0] == 1.0, "Expected output to be 1.0");

    nn_layer_destroy_singly_connected(l);
    return NULL;
}

char* test_nn_layer_activate_singly_connected()
{
    nn_layer_singly_connected_t* l;
    
    l = nn_layer_create_singly_connected(nn_linear_fn, 2);
    l->weights[0] = 1.0;
    l->weights[1] = 1.0;

    float output[2];
    float input[2] = {2, 1};
    nn_layer_activate_singly_connected(l, input, output);
    mu_assert(output[0] == 2.0, "Expected output to be 2.0");
    mu_assert(output[1] == 1.0, "Expected output to be 1.0");

    nn_layer_destroy_singly_connected(l);
    return NULL;
}

char *all_tests() {
    mu_suite_start();

    mu_run_test(test_nn_layer_create_singly_connected);
    mu_run_test(test_nn_layer_aggregate_singly_connected);
    mu_run_test(test_nn_layer_activate_singly_connected);

    return NULL;
}

RUN_TESTS(all_tests)
