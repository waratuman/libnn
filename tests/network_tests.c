#include "minunit.h"
#include "network.h"

char* test_nn_network_create()
{
    nn_layer_t l[1];
    l[0] = *nn_layer_create_connected(linear_activation, sum_of_products_integration, 1, 1);
    nn_network_t* n = nn_network_create(1, l);
    
    mu_assert(n->layerCount == 1, "Layer count == 1");

    nn_network_destroy(n);
    return NULL;
}

char* test_nn_network_activate_1layer()
{
    nn_layer_t l[1];
    l[0] = *nn_layer_create_connected(linear_activation, sum_of_products_integration, 1, 1);
    nn_network_t* n = nn_network_create(1, l);
    
    float input[1] = {1};
    float output[1];

    nn_network_activate(n, input, output);
    mu_assert(output[0] == 0, "Network activation to be 0");

    l[0].weights[0][0] = 1.0;
    nn_network_activate(n, input, output);
    mu_assert(output[0] == 1.0, "Network activation to be 0");

    nn_network_destroy(n);
    return NULL;
}

char* test_nn_network_activate_2layer()
{
    nn_layer_t l[2];
    l[0] = *nn_layer_create_connected(linear_activation, sum_of_products_integration, 1, 1);
    l[1] = *nn_layer_create_connected(linear_activation, sum_of_products_integration, 1, 1);
    nn_network_t* n = nn_network_create(2, l);
    
    float input[1] = {1};
    float output[1];

    nn_network_activate(n, input, output);
    mu_assert(output[0] == 0, "Network activation to be 0");

    l[0].weights[0][0] = 1.0;
    l[1].weights[0][0] = 1.0;
    nn_network_activate(n, input, output);
    mu_assert(output[0] == 1.0, "Network activation to be 0");

    l[0].weights[0][0] = 1.0;
    l[1].weights[0][0] = 0.5;
    nn_network_activate(n, input, output);
    mu_assert(output[0] == 0.5, "Network activation to be 0");

    l[1].activation = sigmoid_activation;
    nn_network_activate(n, input, output);
    char buffer[20];
    sprintf(buffer, "%f", output[0]);
    mu_assert(strcmp(buffer, "0.622459") == 0, "Network activation to be 0");

    nn_network_destroy(n);
    return NULL;
}

char *all_tests() {
    mu_suite_start();

    mu_run_test(test_nn_network_create);
    mu_run_test(test_nn_network_activate_1layer);
    mu_run_test(test_nn_network_activate_2layer);

    return NULL;
}

RUN_TESTS(all_tests);
