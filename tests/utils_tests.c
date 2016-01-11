#include "minunit.h"
#include "utils.h"

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

char *all_tests() {
    mu_suite_start();

    mu_run_test(test_nn_ii2di);
    mu_run_test(test_nn_di2ii);

    return NULL;
}

RUN_TESTS(all_tests);

