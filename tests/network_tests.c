#include "minunit.h"
#include "network.h"

char* test_nn_network_create()
{
    LAYER_TYPE types[1] = {FULLY_CONNECTED};

    nn_layer_fully_connected_t** l = calloc(1, sizeof(nn_layer_fully_connected_t*));
    l[0] = nn_layer_create_fully_connected(linear_activation, sum_of_products_integration, 1, 1);
    nn_network_t* n = nn_network_create(1, types, (void**)l);

    mu_assert(n->layerCount == 1, "Layer count == 1");
    mu_assert(n->error == squared_error, "Default loss == squared_error");

    nn_network_destroy(n);
    return NULL;
}

char* test_nn_network_activate_1layer()
{
    LAYER_TYPE types[1] = {FULLY_CONNECTED};
    nn_layer_fully_connected_t** l = calloc(1, sizeof(nn_layer_fully_connected_t*));
    l[0] = nn_layer_create_fully_connected(linear_activation, sum_of_products_integration, 1, 1);
    nn_network_t* n = nn_network_create(1, types, (void**)l);

    float input[1] = {1};
    float output[1];

    nn_network_activate(n, input, output);
    mu_assert(output[0] == 0, "Network activation to be 0");

    l[0]->weights[0][0] = 1.0;
    nn_network_activate(n, input, output);
    mu_assert(output[0] == 1.0, "Network activation to be 0");

    nn_network_destroy(n);
    return NULL;
}

char* test_nn_network_activate_2layer()
{
    LAYER_TYPE types[2] = {FULLY_CONNECTED, FULLY_CONNECTED};
    nn_layer_fully_connected_t** l = calloc(2, sizeof(nn_layer_fully_connected_t*));
    l[0] = nn_layer_create_fully_connected(linear_activation, sum_of_products_integration, 2, 2);
    l[1] = nn_layer_create_fully_connected(linear_activation, sum_of_products_integration, 2, 1);
    nn_network_t* n = nn_network_create(2, types, (void**)l);

    float input[2] = {1, 1};
    float output[1];

    nn_network_activate(n, input, output);
    mu_assert(output[0] == 0, "Network activation to be 0");

    l[0]->biases[0] = 0;
    l[0]->biases[1] = 0;
    l[1]->biases[0] = 0;

    l[0]->weights[0][0] = 1.0;
    l[0]->weights[0][1] = 1.0;
    l[0]->weights[1][0] = 1.0;
    l[0]->weights[1][1] = 1.0;
    l[1]->weights[0][0] = 0.5;
    l[1]->weights[0][1] = 0.5;
    nn_network_activate(n, input, output);
    mu_assert(output[0] == 2.0, "Network activation to be 2.0");

    l[1]->weights[0][0] = 0.25;
    l[1]->weights[0][1] = 0.25;
    nn_network_activate(n, input, output);
    mu_assert(output[0] == 1, "Network activation to be 1.0");

    l[1]->weights[0][0] = 1;
    l[1]->weights[0][1] = 1;
    l[1]->activation = sigmoid_activation;
    nn_network_activate(n, input, output);
    char buffer[20];
    sprintf(buffer, "%f", output[0]);
    printf("%f", output[0]);
    mu_assert(strcmp(buffer, "0.982014") == 0, "Network activation to be 1 / (1 - e^(-4))");

    nn_network_destroy(n);

    return NULL;
}

char* test_nn_network_loss()
{
    char buffer[20];

    LAYER_TYPE types[1] = {FULLY_CONNECTED};
    nn_layer_fully_connected_t** l = calloc(1, sizeof(nn_layer_fully_connected_t*));
    l[0] = nn_layer_create_fully_connected(linear_activation, sum_of_products_integration, 3, 3);
    nn_network_t* n = nn_network_create(1, types, (void**)l);

    l[0]->weights[0][0] = 1.0;
    l[0]->weights[0][1] = 1.0;
    l[0]->weights[0][2] = 1.0;

    float input[3] = {0, 0, 0};
    float output[3] = {0, 0, 0};
    sprintf(buffer, "%f", nn_network_loss(n, input, output));
    mu_assert(strcmp(buffer, "0.000000") == 0, "Network loss == 0.0");

    input[2] = 1;
    sprintf(buffer, "%f", nn_network_loss(n, input, output));
    mu_assert(nn_network_loss(n, input, output) == (float)1.0 / (float)6.0 , "Network loss == 1 / 6");

    nn_network_destroy(n);

    return NULL;
}

char *all_tests() {
    mu_suite_start();

    mu_run_test(test_nn_network_create);
    mu_run_test(test_nn_network_activate_1layer);
    mu_run_test(test_nn_network_activate_2layer);
    mu_run_test(test_nn_network_loss);

    return NULL;
}

RUN_TESTS(all_tests)
