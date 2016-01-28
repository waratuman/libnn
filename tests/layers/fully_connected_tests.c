#include <nn.h>

#include "minunit.h"

char* test_nn_layer_create_fully_connected()
{
    nn_layer_fully_connected_t* l = nn_layer_create_fully_connected(nn_linear_fn, nn_sop_fn, 1, 1);
    mu_assert(l->inputCount == 1, "inputCount initalize to 1");
    mu_assert(l->outputCount == 1, "outputCount initalize to 1");
    mu_assert(l->activation == nn_linear_fn, "nn_linear_fn activation function");
    mu_assert(l->biases[0] == 0.0, "1 bias");
    mu_assert(l->weights[0][0] == 0.0, "1 weight");
    
    float output[1];
    float input[1] = {1};
    nn_layer_activate_fully_connected(l, input, output);
    mu_assert(output[0] == 0.0, "Expected output to be 0.0");
    
    return NULL;
}

char* test_nn_layer_aggregate_fully_connected()
{
    nn_layer_fully_connected_t* l;
    
    l = nn_layer_create_fully_connected(nn_linear_fn, nn_sop_fn, 2, 1);
    l->weights[0][0] = 1.0;
    l->weights[0][1] = 1.0;

    float output[1];
    float input[2] = {1, 1};
    nn_layer_aggregate_fully_connected(l, input, output);
    mu_assert(output[0] == 2.0, "Expected output to be 1.0");

    return NULL;
}

char* test_nn_layer_activate_fully_connected()
{
    nn_layer_fully_connected_t* l;
    
    l = nn_layer_create_fully_connected(nn_linear_fn, nn_sop_fn, 2, 1);
    l->weights[0][0] = 1.0;
    l->weights[0][1] = 1.0;

    float output[1];
    float input[2] = {1, 1};
    nn_layer_activate_fully_connected(l, input, output);
    mu_assert(output[0] == 2.0, "Expected output to be 1.0");

    return NULL;
}

char *all_tests() {
    mu_suite_start();

    mu_run_test(test_nn_layer_create_fully_connected);
    mu_run_test(test_nn_layer_aggregate_fully_connected);
    mu_run_test(test_nn_layer_activate_fully_connected);

    return NULL;
}

RUN_TESTS(all_tests)
