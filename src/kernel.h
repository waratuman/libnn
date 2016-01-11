#pragma once

/* The kernel offset is the difference between any element and the center of the
kernel. This is usefull for calculating the input into the kernel given the
center input index of the kernel.

For example. A 3x3 kernel would return:
[ [ -1, -1 ], [ -1, 0 ], [ -1, 1 ],
  [  0, -1 ], [  0, 0 ], [  0, 1 ],
   [ 1, -1 ], [  1, 0 ], [  1, 1 ] ]

The result is stored into the result argument, which must have allocated enough
space for all the results, which is the product of each dimesion (3x3 in the case
of the example).

Arguments:
dimensionCount: The number of the dimensions of the kernel
    dimensions: The dimensions of the kernel
        result: A pointer to an array of arrays of ints in which to store the result
*/

void nn_kernel_offset(int dimensionCount, int* dimensions, int** result);

void nn_kernel_center(int dimesionCount, int* dimesions, int* result);
