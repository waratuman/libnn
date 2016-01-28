#include <nn.h>
#include <time.h>

#include "minunit.h"

void pl(nn_layer_type_t lt, void* l) {
    switch(lt) {
        case NN_FC:
            printf("i: %8i [ %i ]\n", nn_layer_input_count(lt, l), nn_layer_input_count(lt, l));
            printf("i: %8i [ %i ]\n", nn_layer_output_count(lt, l), nn_layer_output_count(lt, l));
        break;

        default:
            printf("i: %8i [ ", nn_layer_input_count(lt, l));
            for (int i = 0; i < nn_layer_input_dimension_count(lt, l); i++) {
                printf("%i ", nn_layer_input_dimensions(lt, l)[i]);
            }
            printf("]\no: %8i [ ", nn_layer_output_count(lt, l));
            for (int i = 0; i < nn_layer_output_dimension_count(lt, l); i++) {
                printf("%i ", nn_layer_output_dimensions(lt, l)[i]);
            }
            printf("]\n");
    }
}

char* test_cifar()
{
    int c;
    int* dims;

    // conv1
    int d1[3] = {3, 32, 32};
    int p1[3] = {0, 2, 2};
    int z1[3] = {1, 5, 5};
    int t1[3] = {1, 1, 1};
    nn_layer_convolutional_t* l1 = nn_layer_create_convolutional(nn_linear_fn, nn_sop_fn, 3072, 3, 32, d1, p1, t1, z1);
    printf("conv1\n");
    pl(NN_CV, l1);

    // pool1 / relu1
    int d2[4] = {32, 3, 32, 32};
    int p2[4] = {0, 0, 0, 0};
    int z2[4] = {1, 1, 3, 3};
    int t2[4] = {1, 1, 2, 2};
    nn_layer_convolutional_t* l2 = nn_layer_create_convolutional(nn_relu_fn, nn_max_fn, nn_layer_output_count(NN_CV, l1), 4, 1, d2, p2, t2, z2);
    printf("pool1 / relu1\n");
    pl(NN_CV, l2);

    // norm1
    c = nn_layer_output_dimension_count(NN_CV, l2);
    dims = calloc(c - 1, sizeof(int));
    for (int i = 1; i < c; i++) {
        dims[i - 1] = nn_layer_output_dimensions(NN_CV, l2)[i];
    }
    int z3[4] = {1, 1, 3, 3};
    nn_layer_lrn_t* l3 = nn_layer_create_lrn(nn_layer_output_count(NN_CV, l2), c - 1, dims, z3, 2, 0.00005, 0.75);
    printf("norm1\n");
    pl(NN_LRN, l3);
    free(dims);

    // conv2 / relu2
    int d4[4] = {32, 3, 15, 15};
    int p4[4] = {0, 0, 2, 2};
    int z4[4] = {1, 1, 5, 5};
    int t4[4] = {1, 1, 1, 1};
    printf("conv2 / relu2\n");
    nn_layer_convolutional_t* l4 = nn_layer_create_convolutional(nn_relu_fn, nn_sop_fn, nn_layer_output_count(NN_CV, l3), 4, 32, d4, p4, t4, z4);
    pl(NN_CV, l4);

    // pool2
    int d5[5] = {32, 32, 3, 15, 15};
    int p5[5] = {0, 0, 0, 0, 0};
    int z5[5] = {1, 1, 1, 3, 3};
    int t5[5] = {1, 1, 1, 2, 2};
    nn_layer_convolutional_t* l5 = nn_layer_create_convolutional(nn_linear_fn, nn_avg_fn, nn_layer_output_count(NN_CV, l4), 5, 1, d5, p5, t5, z5);
    printf("pool2\n");
    pl(NN_CV, l5);

    // norm2
    c = nn_layer_output_dimension_count(NN_CV, l5);
    dims = calloc(c - 1, sizeof(int));
    for (int i = 1; i < c; i++) {
        dims[i - 1] = nn_layer_output_dimensions(NN_CV, l5)[i];
    }
    int z6[6] = {1, 1, 1, 1, 3, 3};
    nn_layer_lrn_t* l6 = nn_layer_create_lrn(nn_layer_output_count(NN_CV, l5), c - 1, dims, z6, 2, 0.00005, 0.75);
    printf("norm2\n");
    pl(NN_LRN, l6);
    free(dims);

    // conv3 / relu3
    int d7[5] = { 32, 32, 3, 7, 7 };
    int p7[5] = { 0, 0, 0, 2, 2 };
    int z7[5] = { 1, 1, 1, 5, 5 };
    int t7[5] = { 1, 1, 1, 1, 1 };
    nn_layer_convolutional_t* l7 = nn_layer_create_convolutional(nn_relu_fn, nn_sop_fn, nn_layer_output_count(NN_CV, l6), 5, 64, d7, p7, t7, z7);
    printf("conv3\n");
    pl(NN_CV, l7);

    // pool3
    int d8[6] = { 64, 32, 32, 3, 7, 7 };
    int p8[6] = { 0, 0, 0, 0, 0, 0 };
    int z8[6] = { 1, 1, 1, 1, 3, 3 };
    int t8[6] = { 1, 1, 1, 1, 2, 2 };
    nn_layer_convolutional_t* l8 = nn_layer_create_convolutional(nn_linear_fn, nn_avg_fn, nn_layer_output_count(NN_CV, l7), 6, 1, d8, p8, t8, z8);
    printf("pool3\n");
    pl(NN_CV, l8);

    // ip1
    nn_layer_fully_connected_t* l9 = nn_layer_create_fully_connected(nn_linear_fn, nn_sop_fn, nn_layer_output_count(NN_CV, l8), 10);
    printf("ip1\n");
    pl(NN_FC, l9);

    // void** ls = calloc(9, sizeof(void*));
    // ls[0]= l1;
    
    nn_layer_type_t lt[9] = {NN_CV, NN_CV, NN_LRN, NN_CV, NN_CV, NN_LRN, NN_CV, NN_CV, NN_FC};
    void* ls[9] = {l1, l2, l3, l4, l5, l6, l7, l8, l9};
    nn_network_t* n = nn_network_create(9, lt, ls);


    uint8_t intInput[3072];
    FILE* f = fopen("tests/data/cifar-10-batches-bin/data_batch_1.bin", "r");
    fread(intInput, sizeof(uint8_t), 1, f);
    fread(intInput, sizeof(uint8_t), 3072, f);
        
    float input[3072];
    float output[10];

    for (int i = 0; i < 3072; i++) {
        input[i] = (float)intInput[i];
    };

    clock_t start, stop;
    start = clock();
    nn_network_activate(n, input, output);
    stop = clock();

    for (int i = 0; i < 10; i++) {
        printf("%f ", output[i]);
    }
    printf("\n%f s\n", (double)(stop - start) / CLOCKS_PER_SEC);

    nn_network_destroy(n);
    
    return NULL;
}

char *all_tests() {
    mu_suite_start();

    mu_run_test(test_cifar);

    return NULL;
}

RUN_TESTS(all_tests)
