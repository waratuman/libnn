#include "minunit.h"
#include "utils.h"
#include "layer.h"
#include "layers/loss.h"

char* test_nn_layer_init_loss()
{
    nn_layer_loss_t* l = calloc(1, sizeof(nn_layer_loss_t));
    nn_layer_init_loss(l);
    return NULL;
}

char* test_nn_layer_create_loss()
{
    nn_layer_loss_t* l = nn_layer_create_loss(sum_of_squares_integration, 3);
    mu_assert(l->inputCount == 3, "Input count == 3");
    return NULL;
}

char* test_nn_layer_activate_loss()
{
    nn_layer_loss_t* l = nn_layer_create_loss(sum_of_squares_integration, 3);
    
    float input[3] = {0, 0, 0};
    float output[3] = {0, 0, 0};
    float a = nn_layer_activate_loss(l, input, output);
    mu_assert(a == 0, "Loss == 0");

    input[2] = 1;
    a = nn_layer_activate_loss(l, input, output);
    mu_assert(a == (float)1.0 / (float)6.0 , "Loss == 0");

    return NULL;
}

char* all_tests() {
    mu_suite_start();

    mu_run_test(test_nn_layer_init_loss);
    mu_run_test(test_nn_layer_create_loss);
    mu_run_test(test_nn_layer_activate_loss);

    return NULL;
}

RUN_TESTS(all_tests);
