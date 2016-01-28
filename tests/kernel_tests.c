#include "minunit.h"
#include "utils.h"

char* test_nn_kernel_offset()
{
    int dimensionCount = 3;
    int dimensions[3] = {3, 3, 3};

    int** result = calloc(3 * 3 * 3, sizeof(int*));
    for (int i = 0; i < 3 * 3 * 3; i++) {
        result[i] = calloc(dimensionCount, sizeof(int));
    }

    nn_kernel_offset(dimensionCount, dimensions, result);

    for (int i = 0; i < 27; i++) {
        mu_assert(result[i][0] == -1 + i / 9, "Error in nn_kernel_offset");
        mu_assert(result[i][1] == -1 + i % 9 / 3, "Error in nn_kernel_offset");
        mu_assert(result[i][2] == -1 + i % 3, "Error in nn_kernel_offset");
    }

    return NULL;
}

char *all_tests() {
    mu_suite_start();

    mu_run_test(test_nn_kernel_offset);

    return NULL;
}

RUN_TESTS(all_tests)
