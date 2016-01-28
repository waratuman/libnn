#include <nn.h>

#include "minunit.h"

char* test_nn_layer_init_lrn()
{
    nn_layer_lrn_t* l = calloc(1, sizeof(nn_layer_lrn_t));
    nn_layer_init_lrn(l);
    mu_assert(l->outputCount == 0, "outputCount initalize to 0");

    int dim[2] = {5, 5};
    int size[2] = {3, 3};
    int odim[3] = {2, 0, 0};
    l->kernelInputCount = 2;
    l->inputCount = 25;
    l->inputDimensionCount = 2;
    l->inputDimensions = dim;
    l->outputDimensions = odim;
    l->size = size;
    nn_layer_init_lrn(l);

    mu_assert(l->outputCount == 25, "outputCount initalize to 25");
    mu_assert(l->outputDimensions[0] == 5, "outputCount initalize to 5");
    mu_assert(l->outputDimensions[1] == 5, "outputCount initalize to 5");

    return NULL;
}

char* test_nn_layer_create_lrn()
{
    int dim[3] = {5, 5, 2};
    int siz[3] = {5, 5, 1};

    nn_layer_lrn_t* l = nn_layer_create_lrn(50, 3, dim, siz, 2, 0.0001, 0.75);

    mu_assert(l->inputCount == 50, "inputCount init to 25");
    mu_assert(l->outputCount == 50, "outputCount init to 50");
    mu_assert(l->inputDimensionCount == 3, "dimensionCount init to 3");

    for (int i = 0; i < 3; i++) {
        mu_assert(l->inputDimensions[i] == dim[i], "inputDimensions to be intialized");
        mu_assert(l->outputDimensions[i] == dim[i], "outputDimensions to be equal to inputDimensions");
        mu_assert(l->size[i] == siz[i], "Size to be intialized");
    }

    return NULL;
}

char* test_nn_layer_activate_lrn()
{
    char buffer[20];
    int d1[2] = {3,3};
    int s1[2] = {3,3};

    nn_layer_lrn_t* l1 = nn_layer_create_lrn(9, 2, d1, s1, 2, 0.0001, 0.75);

    float input[9] = { 0, 1, 0,
                       1, 0, 1,
                       0, 1, 0 };

    float output[l1->outputCount];
    nn_layer_activate_lrn(l1, input, output);

    for (int i = 0; i < 9; i++) {
        sprintf(buffer, "%f", output[i]);
        if (i % 2 == 1) {
            mu_assert(strcmp(buffer, "0.594537") == 0, "Layer activation to be 1 / (2 + 10^-4 * 3)^0.75");
        } else {
            mu_assert(strcmp(buffer, "0.000000") == 0, "Layer activation to be 0 / (2 + 10^-4 * 3)^0.75");
        }
    }

    return NULL;
}

char* test_nn_layer_lrn_is_index_padding()
{
    int d1[2] = {2,2};
    int s1[2] = {2,2};

    nn_layer_lrn_t* l1 = nn_layer_create_lrn(9, 2, d1, s1, 2, 0.0001, 0.75);

    int in[2] = { -1, -1 };
    mu_assert(nn_layer_is_lrn_index_padding(l1, in) == true, "Expected index to be included in padding");

    in[0] = -1;
    in[1] = 0;
    mu_assert(nn_layer_is_lrn_index_padding(l1, in) == true, "Expected index to be included in padding");

    in[0] = 0;
    in[1] = -1;
    mu_assert(nn_layer_is_lrn_index_padding(l1, in) == true, "Expected index to be included in padding");

    in[0] = 0;
    in[1] = 0;
    mu_assert(nn_layer_is_lrn_index_padding(l1, in) == false, "Expected index to be included in padding");

    in[0] = 2;
    in[1] = 2;
    mu_assert(nn_layer_is_lrn_index_padding(l1, in) == true, "Expected index to be included in padding");

    in[0] = 2;
    in[1] = 1;
    mu_assert(nn_layer_is_lrn_index_padding(l1, in) == true, "Expected index to be included in padding");

    in[0] = 1;
    in[1] = 2;
    mu_assert(nn_layer_is_lrn_index_padding(l1, in) == true, "Expected index to be included in padding");

    in[0] = 1;
    in[1] = 1;
    mu_assert(nn_layer_is_lrn_index_padding(l1, in) == false, "Expected index to be included in padding");

    return NULL;
}

char *all_tests() {
    mu_suite_start();

    mu_run_test(test_nn_layer_init_lrn);
    mu_run_test(test_nn_layer_create_lrn);
    mu_run_test(test_nn_layer_activate_lrn);
    mu_run_test(test_nn_layer_lrn_is_index_padding);

    return NULL;
}

RUN_TESTS(all_tests)
